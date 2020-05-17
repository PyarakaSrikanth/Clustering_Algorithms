def build_model(cluserting_model, data, labels):
    
	model = clustering_model(data)
	
	print("homo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
	print(50 * "-")
	
	print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t'
	       %(metrics.homogeneity_score(labels, model.labels_),
		   %(metrics.completeness_score(labels,model.labels_),
		   %(metrics.v_measure_score(labels,model.labels_),
		   %(metrics.adjusted_rand_score(labels,model.labels_),
		   %(metrics.adjusted_mutual_info_score(labels,labels_),
		   %(metrics.silhouette(data,model.labels_)))
		   

def k_means(data,n_clusters=3,max_iter=1000):
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter).fit(data)
    return model 

	