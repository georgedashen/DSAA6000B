CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 3 --t 0.1 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
	CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 3 --t 0.3 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
	CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 3 --t 0.5 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
	CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 3 --t 0.7 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
	CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 3 --t 0.9 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
	CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 5 --t 0.1 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
	CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 5 --t 0.3 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
        CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 5 --t 0.5 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
        CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 5 --t 0.7 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
        CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 5 --t 0.9 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
	CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 7 --t 0.1 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
        CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 7 --t 0.3 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
        CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 7 --t 0.5 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
        CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 7 --t 0.7 --simMat cosine_sim_mat_diag0.pickle --epoch 750 &&
        CUDA_VISIBLE_DEVICES=7 python evaluate_knn.py --k 7 --t 0.9 --simMat cosine_sim_mat_diag0.pickle --epoch 750

