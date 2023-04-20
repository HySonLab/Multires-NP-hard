import torch

'''
This function generate centroid lists by minimize cost in each order
'''
def generate_centroids_via_total_distance_per_order(self,similarity_maxtrix,batched_order=4):
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
def cluster_via_merge_sort(self,similar_matrix,batched_order=4):
    size=len(similar_matrix.keys())
    num_clusters=int(size/batched_order)
    similar_matrix_torch=torch.zeros(size,size)
    for i in range(similar_matrix_torch.shape[0]):
        for j in range(similar_matrix_torch.shape[0]):
            similar_matrix_torch[i][j]=round(similar_matrix[str(i+1)][str(j+1)],2)   
    list_centroids=self.generate_centroids_via_total_distance_per_order(similar_matrix_torch.detach().clone())
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