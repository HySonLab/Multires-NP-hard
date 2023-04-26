import torch

def create_similarity_matrix(input):
  result=torch.cdist(input, input, p=2.0)
  return result
'''
This function generate centroid lists by minimize cost in each order
'''
def generate_centroids_via_total_distance_per_order(similarity_maxtrix,batched_order=4):
    matrix_shape=similarity_maxtrix.shape[0]
    number_clusters=int(matrix_shape/batched_order)
    similar_sum=torch.sum(similarity_maxtrix,axis=0)
    _,indices=torch.sort(similar_sum)
    list_centroids=[]
    for i in range(number_clusters):
        list_centroids.append(indices[i].item())
    return list_centroids
    
'''
This function cluster all the item from generated centroids via merger sort
'''
def cluster_via_merge_sort(similar_matrix_torch,batched_order=4):
    size=similar_matrix_torch.shape[0]
    num_clusters=int(size/batched_order)
    # similar_matrix_torch=torch.zeros(size,size)
    # for i in range(similar_matrix_torch.shape[0]):
    #     for j in range(similar_matrix_torch.shape[0]):
    #         similar_matrix_torch[i][j]=round(similar_matrix[str(i+1)][str(j+1)],2)   
    list_centroids=generate_centroids_via_total_distance_per_order(
        similar_matrix_torch.detach().clone(),batched_order=batched_order)
    similar_centroids=similar_matrix_torch[list_centroids]
    sorted_similar_centroids,indices=torch.sort(similar_centroids,axis=1)
    remained_item=list(list_centroids)
    list_clusters=[]
    cluster_column=[]
    for i in range(num_clusters):
        list_clusters.append([list_centroids[i]])
        cluster_column.append(1)
    while(len(remained_item)<size):
        current_list=[]
        number_cluster=[]
        for i in range(len(list_clusters)):
            number_cluster.append(len(list_clusters[i]))
        for i in range(len(cluster_column)):
            current_list.append(sorted_similar_centroids[i][cluster_column[i]])
        selected_row=current_list.index(min(current_list))
        max_index_y=cluster_column[selected_row]
        max_index=indices[selected_row][max_index_y].item()
        if(max_index not in remained_item):
            remained_item.append(max_index)
            cluster_column[selected_row]+=1
            list_clusters[selected_row].append(max_index)
            if(len(list_clusters[selected_row])==4):
                sorted_similar_centroids[selected_row]=100000
        else:
            cluster_column[selected_row]+=1

    return list_clusters

# print("test")
# data=torch.rand(32,20,2)
# similar=create_similarity_matrix(data)
# result=[]
# for i in range(similar.shape[0]):
#     current_data=similar[i]
#     length=current_data.shape[0]
#     clustered=cluster_via_merge_sort(current_data,batched_order=5)
#     reduced_x=torch.rand(4,2)
#     for i in range(len(clustered)):
#         selected_data=current_data[clustered[i]]
#         reduced_x[i][0]=torch.sum(selected_data[:,0])/length
#         reduced_x[i][1]=torch.sum(selected_data[:,1])/length
# print(reduced_x)
# my_data=torch.Tensor([[1,2],[3,4],[5,6]])
# print
# data=torch.sum(my_data[:,0])
# data=torch.sum(my_data[:,1])
# print(data)