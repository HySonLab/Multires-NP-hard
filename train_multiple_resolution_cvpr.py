import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
from utils.cluster import *

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline,baseline_1, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts,previous_cost):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    print(" previous reward " + str( previous_cost))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model[0].train()
    model[1].train()
    set_decode_type(model[0], "sampling")
    set_decode_type(model[1], "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            baseline_1,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    avg_reward = validate(model[0], val_dataset, opts)
    if(avg_reward<previous_cost or previous_cost==0):

    #if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model[0]).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}-cost-{}.pt'.format(epoch,avg_reward))
        )

    #avg_reward = validate(model[0], val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model[0], epoch)
    baseline_1.epoch_callback(model[1], epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    return avg_reward


def train_batch(
        model_lists,
        optimizer,
        baseline,
        baseline_1,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    #similar_matrix=create_similarity_matrix(x)
    # Evaluate model, get costs and log probabilities
    #print(x)
    
    graph_size=x['loc'].shape[1]
    ori_start_merge=5
    loss_list=[]
    ###################################
    #data=torch.rand(32,20,2)
    similar=create_similarity_matrix(x['loc'])
    demand=x['demand']
    print(similar.shape)
    level=0
    start_merge=ori_start_merge**level
    while(graph_size/start_merge>3):
        reduced_x=torch.rand(x['loc'].shape[0],int(x['loc'].shape[1]/start_merge),2)
        reduced_demand=torch.rand(x['loc'].shape[0],int(x['loc'].shape[1]/start_merge))
        #print(reduced_x)
        for i in range(similar.shape[0]):
            current_data=similar[i]
            clustered=cluster_via_merge_sort(current_data,batched_order=start_merge)
            current_x=x['loc'][i]
            current_demand=x['demand'][i]
            for j in range(len(clustered)):
                selected_data=current_x[clustered[j]]
                selected_demand=current_demand[clustered[j]]
                reduced_x[i][j][0]=torch.sum(selected_data[:,0])/selected_data.shape[0]
                reduced_x[i][j][1]=torch.sum(selected_data[:,1])/selected_data.shape[0]
                reduced_demand[i][j]=torch.sum(selected_demand)/selected_demand.shape[0]
        #original cost
        new_x=dict()
        new_x['loc']=reduced_x
        new_x['demand']=reduced_demand
        new_x["depot"]=reduced_x[:,0,:]
        print(new_x['loc'].shape)
        print(new_x['demand'].shape)
        print(new_x['depot'].shape)
        reduced_x=move_to(new_x,opts.device)
        #print(level,start_merge)
        cost, log_likelihood = model_lists[0](reduced_x)
        
        #print(cost)
        # Evaluate baseline, get baseline loss if any (only for critic)
        #if(level==0):
        bl_val, bl_loss = baseline.eval(reduced_x, cost) if bl_val is None else (bl_val, 0)
        # else:
        #     bl_val, bl_loss = baseline_1.eval(reduced_x, cost) if bl_val is None else (bl_val, 0)
        # Calculate loss
        reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
        loss_current=reinforce_loss + bl_loss
        loss_list.append( reinforce_loss + bl_loss)
        level+=1
        start_merge=ori_start_merge**level
    loss=sum(loss_list)/len(loss_list)
    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)


print("test")