import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance
from PIL import Image
import cv2
np.set_printoptions(threshold=np.inf, suppress=False)
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

import check_utils


def set_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=10)
    parser.add_argument('--scale_down_times', type=int, default=3)
    parser.add_argument('--scale_down_loop', type=int, default=0)
    parser.add_argument('--k', type=int, default=10, help='used in k-NN')

    parser.add_argument('--image', type=str, default='2') # 1, 2
    parser.add_argument('--init_method', type=str, default='color_cloest_picked') # ...
    parser.add_argument('--gamma_s', type=float, default=0.1) # 10, 0.1, 0.001
    parser.add_argument('--gamma_c', type=float, default=0.1)
    parser.add_argument('--gif_loop', type=int, default=0)
    parser.add_argument('--cut_type', type=str, default='ratio_cut') # *2
    # parser.add_argument('--cut_type', type=str, default='normalized_cut')
    args = parser.parse_args()

    return args


def scale_down_img(img, method, times, loop):
    if method == 'pyr':
        for _ in range(times):
            img = cv2.pyrDown(img)
    elif loop != 0:
        for _ in range(times):
            W, H = (np.ceil(i / 2).astype(np.int) for i in img.shape)
            img = cv2.resize(img, dsize=(H, W), interpolation=method)
    else:
        W, H = (np.ceil(i / 2 ** times).astype(np.int) for i in img.shape)
        img = cv2.resize(img, dsize=(H, W), interpolation=method)

    return img


def scale_up_img(img, method, times, loop):
    if method == 'pyr':
        for _ in range(times):
            img = cv2.pyrUp(img)
    elif loop != 0:
        for _ in range(times):
            W, H = (np.ceil(i * 2).astype(np.int) for i in img.shape)
            img = cv2.resize(img, dsize=(H, W), interpolation=method)
    else:
        W, H = (np.ceil(i * 2 ** times).astype(np.int) for i in img.shape)
        img = cv2.resize(img, dsize=(H, W), interpolation=method)

    return img


def get_data(path='Yale_Face_Database', subject_num=15, scale_setting={}):
    face_expression = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

    scale_down_method = scale_setting.get('scale_down_method', 'pyr')
    scale_down_times = scale_setting.get('scale_down_times', 2)
    scale_down_loop = scale_setting.get('scale_down_loop', 0)

    X_train, y_train, X_test, y_test = [], [], [], []
    for subject in range(1, subject_num + 1):
        for expression in face_expression:
            filename = 'subject%02d.%s.pgm' % (subject, expression)
            training_path = os.path.join(path, 'Training', filename)
            testing_path = os.path.join(path, 'Testing', filename)
            if os.path.isfile(training_path):
                img = scale_down_img(plt.imread(training_path),
                                     method=scale_down_method, times=scale_down_times,
                                     loop=scale_down_loop)
                X_train.append(img)
                y_train.append(subject-1)
            elif os.path.isfile(testing_path):
                img = scale_down_img(plt.imread(testing_path),
                                     method=scale_down_method, times=scale_down_times,
                                     loop=scale_down_loop)
                X_test.append(img)
                y_test.append(subject-1)

                """
                img1 = scale_down_img(plt.imread(testing_path))
                img2 = scale_down_img(plt.imread(testing_path), method=cv2.INTER_CUBIC)
                img3 = scale_down_img(plt.imread(testing_path), method=cv2.INTER_CUBIC, cons=True)
                print(img1.shape)
                print(img2.shape)
                print(img3.shape)
                check_utils.show_three_imgs([img1, img2, img3])
                """

    TRAIN_NUM = len(X_train)
    TEST_NUM = len(X_test)
    X_train = np.array(X_train)
    shape = X_train[0].shape
    X_train = X_train.reshape(TRAIN_NUM, -1)
    y_train = np.array(y_train)
    X_test = np.array(X_test).reshape(TEST_NUM, -1)
    y_test = np.array(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # check_utils.check_resize_resolution()

    return X_train, y_train, X_test, y_test, shape


def PCA(X_train, kernel_setting, img_shape, embedding_dim, eigenfaces_num=25):
    """
    In PCA, we want to find a orthonormal projection
    by solving the eigenevalue problem of covaraince of training data
    """
    H, W = img_shape

    gram_matrix = get_kernel(X_train, kernel_setting)
    print('PCA Gram:', gram_matrix.shape)

    eigen_values, eigen_vectors = np.linalg.eig(gram_matrix)
    print('Eigen-values:', eigen_values.shape)
    print('Eigen-vectors:', eigen_vectors.shape)

    descending_index = np.argsort(eigen_values)[::-1]
    descending_eigen_vectors = eigen_vectors[:, descending_index]
    # check descending
    # print(eigen_values[descending_index][:10])

    projection_matrix = descending_eigen_vectors[:, :embedding_dim]
    projection_matrix = np.real(projection_matrix)
    print('projection_matrix', projection_matrix.shape)
    # check non-complex number
    # print(np.sum(np.imag(projection_matrix)))

    eigen_faces = None
    if kernel_setting['kernel_type'] == 'covariance':
        eigen_faces = descending_eigen_vectors[:, :eigenfaces_num].reshape(H, W, -1)
        eigen_faces = np.real(eigen_faces)
        eigen_faces = np.transpose(eigen_faces, (2, 0, 1))
        print('faces:', eigen_faces.shape)

    return projection_matrix, eigen_faces


def classify_subject_normal(X_train, y_train, X_test, y_test, projection_matrix, k):
    """
    Implement k-NN on what embedding space
    """
    print('X_train', X_train.shape)
    print('projection matrix', projection_matrix.shape)
    z_train = X_train.dot(projection_matrix)
    print('z_train', z_train.shape)
    z_test = X_test.dot(projection_matrix)
    print('z_test', z_test.shape)


    k_NN(z_train, y_train, z_test, y_test, k)


def classify_subject_kernel_pca(X_train, y_train, X_test, y_test, projection_matrix, k, kernel_setting):
    X = np.concatenate([X_train, X_test])
    gram_matrix = get_kernel(X, kernel_setting)
    print('X', X.shape)
    print('gram in classify', gram_matrix.shape)
    print('projection matrix', projection_matrix.shape)
    z_train = np.empty((projection_matrix.shape))
    z_test = np.empty((X_test.shape[0], projection_matrix.shape[1]))
    print('z_train', z_train.shape)
    print('z_test', z_test.shape)

    """
    projection_matrix = projection_matrix.T
    M = projection_matrix.shape[0]
    N = X_train.shape[0]
    new_N = X_test.shape[0]
    for now in range(new_N): # new data, which index >= 135
        for k in range(M): # embedding dimension of each data
            sumit = 0
            for i in range(N):
                sumit += projection_matrix[k, i] * gram_matrix[now+135, i]
            z_test[now, k] = sumit

    M = projection_matrix.shape[0]
    N = X_train.shape[0]
    new_N = X_train.shape[0]
    for now in range(new_N): # new data, which index >= 135
        for k in range(M): # embedding dimension of each data
            sumit = 0
            for i in range(N):
                sumit += projection_matrix[k, i] * gram_matrix[now, i]
            z_train[now, k] = sumit


    projection_matrix = projection_matrix.T
    z_train = np.empty((projection_matrix.shape))
    z_test = np.empty((X_test.shape[0], projection_matrix.shape[1]))
    for new_x in range(X_train.shape[0]):
        z_train[new_x] = gram_matrix[new_x, :135].dot(projection_matrix)

    for new_x in range(X_test.shape[0]):
        z_test[new_x] = gram_matrix[new_x+135, :135].dot(projection_matrix)
    """

    z_train = gram_matrix[:135, :135].dot(projection_matrix)
    z_test = gram_matrix[135:, :135].dot(projection_matrix)
    print('-'*30)
    print(gram_matrix[0, :135].shape)
    print(projection_matrix.shape)
    print('-'*30)
    k_NN(z_train, y_train, z_test, y_test, k)
    raise NotImplementedError


def k_NN(z_train, y_train, z_test, y_test, k):
    classify = distance.cdist(z_test, z_train, metric='sqeuclidean')
    print('classify', classify.shape)
    # 最小距離的 index，認為跟哪個 z_train 最接近
    sort_index = np.argsort(classify, axis=1) # 30, 135
    print('sort index', sort_index.shape)

    # k-NN
    # 找算完距離的結果的那些 z_train 的真實 label 是什麼
    # y_test * y_trains
    extended_y = np.tile(y_train, reps=(len(y_test), 1)) # 30, 135
    # 找到真實 label 了，並經過排序了
    sorted_y = np.array([arr[sort_index[i]] for i, arr in enumerate(extended_y)])
    # different size
    # count first k elements
    # 算前 k 個誰出現最多當作分類結果 (所以算法要依賴 train 的 label)
    counts = np.array([np.bincount(arr[:k]) for arr in sorted_y])
    # the most frequent as result
    result = np.array([np.argmax(arr) for arr in counts])
    print(sorted_y[:5, :6])

    print('result', result.shape)
    print(result)
    print(y_test)

    # print(y_train[result])
    # print('Performance:', np.mean(y_train[result] == y_test))
    print('Performance:', np.mean(result == y_test))


def save_imgs(imgs, title):
    raise NotImplementedError
    # resize


def get_kernel(X_train, kernel_setting={}):
    kernel_type = kernel_setting.get('kernel_type', 'covariance')
    gamma = kernel_setting.get('gamma', 1)
    coef0 = kernel_setting.get('coef0', 0)
    degree = kernel_setting.get('degree', 2)

    print('Use kernel:', kernel_type)
    if kernel_type == 'covariance':
        mean = np.mean(X_train, axis=0, keepdims=True)
        kernel = np.cov((X_train - mean).T)
    else:
        if kernel_type == 'linear':
            kernel = np.dot(X_train, X_train.T)
        elif kernel_type == 'RBF':
            kernel = np.exp(-gamma * distance.cdist(X_train, X_train, metric='sqeuclidean'))
        elif kernel_type == 'polynomial':
            kernel = (gamma * np.dot(X_train, X_train.T) + coef0) ** degree
            # from sklearn.metrics.pairwise import polynomial_kernel
            # kernel = polynomial_kernel(HI, gamma=gamma, degree=degree, coef0=coef0)
            # print('WT', np.sum(np.abs(kernel_ - kernel)))
            # print(kernel[0, 0])
            # print('me', kernel_[0, 0])
            print(kernel.dtype)
        elif kernel_type == 'tanh':
            kernel = np.tanh(np.dot(X_train, X_train.T) + coef0)

        N = X_train.shape[0]
        matrix_1_N = np.ones((N, N)) / N
        # center the data in feature space
        left_dot = matrix_1_N.dot(kernel)
        kernel = kernel - 2 * left_dot + left_dot.dot(matrix_1_N)
        # kernel = kernel - left_dot - kernel.dot(matrix_1_N) + left_dot.dot(matrix_1_N)
        # print('ERROR:', np.sum(less_kernel - kernel))

    return kernel


def get_class_mean(X_train, y_train):
    feature_size = X_train.shape[1]
    class_num = np.max(y_train) + 1

    class_mean = np.empty((class_num, feature_size))
    for i in range(class_num):
        class_mean[i] = np.mean(X_train[y_train == i], axis=0)

    mean = np.mean(X_train, axis=0)
    return mean, class_mean


def get_within_matrix(X_train, y_train, class_mean):
    # p.179
    feature_size = X_train.shape[1]
    class_num = np.max(y_train) + 1

    within = np.zeros((feature_size, feature_size))
    for i in range(class_num):
        dis_to_center = X_train[y_train == i] - class_mean[i]
        within += np.dot(dis_to_center.T, dis_to_center)

    return within


def get_between_matrix(mean, class_mean, y_train):
    # p.179
    feature_size = X_train.shape[1]
    class_num = np.max(y_train) + 1
    unique, counts = np.unique(y_train, return_counts=True)
    dis_to_center = class_mean - mean
    between = np.dot(dis_to_center.T, counts[:, None] * dis_to_center)

    return between


def LDA(X_train, y_train, img_shape, embedding_dim, fisherfaces_num=25):
    H, W = img_shape

    mean, class_mean = get_class_mean(X_train, y_train)
    within = get_within_matrix(X_train, y_train, class_mean)
    between = get_between_matrix(mean, class_mean, y_train)

    # within_class inv usually not exists
    eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.linalg.pinv(within), between))

    descending_index = np.argsort(eigen_values)[::-1]
    descending_eigen_vectors = eigen_vectors[:, descending_index]

    projection_matrix = descending_eigen_vectors[:, :embedding_dim]
    # There are no imaginary numbers if the #data is bigger than the dim
    projection_matrix = np.real(projection_matrix)

    fisherfaces = None
    # if kernel_setting['kernel_type'] == 'covariance':
    fisherfaces = descending_eigen_vectors[:, :fisherfaces_num].reshape(H, W, -1)
    fisherfaces = np.real(fisherfaces)
    fisherfaces = np.transpose(fisherfaces, (2, 0, 1))

    return projection_matrix, fisherfaces


def kernel_LDA(X_train, y_train, embedding_dim, kernel_setting):
    C = np.zeros((X_train.shape[0], len(np.unique(y_train))))
    print('C: ', C.shape) # 135, 15
    for idx, j in enumerate(np.unique(y_train)):
        C[y_train == j,idx] = 1;

    # print(C)
    kernel = get_kernel(X_train, kernel_setting)

    # get class_mean and mean such like in LDA
    print('np.dot(kernel.T, C)', kernel.T.shape, C.shape, np.dot(kernel.T, C).shape)
    # class_mean
    class_mean = np.matmul(kernel.T, C) / np.sum(C, axis = 0) # class_mean = Mj
    print('class_mean:', class_mean.shape)
    # mean
    mean = np.mean(kernel, axis=0)
    print('mean:', mean.shape)

    dis_to_center = class_mean - mean[:, None]

    between = np.matmul(dis_to_center * np.sum(C, axis = 0), dis_to_center.T)

    W = kernel.T - np.matmul(class_mean, C.T)
    within = np.zeros(between.shape)
    for group in np.unique(y_train):
        w = W[:, y_train == group]
        within += (np.dot(w, w.T) / w.shape[1])

    eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.linalg.pinv(within), between))

    # eigen_values, eigen_vectors = np.linalg.eig(gram_matrix)

    descending_index = np.argsort(eigen_values)[::-1]
    descending_eigen_vectors = eigen_vectors[:, descending_index]

    # TODO 1:k+1?
    # These vectors are float64 type
    projection_matrix = descending_eigen_vectors[:, :embedding_dim]
    projection_matrix = np.real(projection_matrix)

    return np.matmul(kernel, projection_matrix), projection_matrix


if __name__ == '__main__':
    args = set_argument()

    # get scale-down input image
    scale_setting = {}
    # scale_setting['scale_down_method'] = 'pyr' # cv2.INTER_CUBIC
    # this method is better TODO
    scale_setting['scale_down_method'] = cv2.INTER_CUBIC
    scale_setting['scale_down_times'] = args.scale_down_times
    scale_setting['scale_down_loop'] = args.scale_down_loop
    X_train, y_train, X_test, y_test, img_shape = get_data(scale_setting=scale_setting)

    SHOW_FACES = False
    SHOW_RECONS_FACES = False

    # LDA
    # projection_matrix, fisherfaces = LDA(X_train, y_train, img_shape, args.embedding_dim)
    # print('projection_matrix:', projection_matrix.shape)
    # print('fisherfaces:', fisherfaces.shape)
    if SHOW_FACES:
        num = fisherfaces.shape[0]
        for i in range(num):
            plt.subplot(5, 5, i+1)
            plt.imshow(scale_up_img(fisherfaces[i], cv2.INTER_CUBIC, args.scale_down_times, args.scale_down_loop), cmap='gray')
            plt.axis('off')
        plt.show()

    if SHOW_RECONS_FACES:
        reconstructed_faces = X_train.dot(projection_matrix.dot(projection_matrix.T)).reshape(-1, *img_shape)
        print('reconstructed_faces', reconstructed_faces.shape)
        shown_recons_faces = reconstructed_faces[np.random.randint(len(reconstructed_faces), size=10)]
        for i in range(5):
            plt.subplot(2, 5, i+1)
            plt.imshow(scale_up_img(shown_recons_faces[i], cv2.INTER_CUBIC, args.scale_down_times, args.scale_down_loop), cmap='gray')
            plt.axis('off')
        plt.show()
        print('shown_recons_faces', shown_recons_faces.shape)

    # 0.9333
    # dim=30, k=10, scale=3 -> 100%
    # classify_subject_normal(X_train, y_train, X_test, y_test, projection_matrix, args.k)

    CONVENIENT_TYPE = 'RBF'
    kernel_setting = {'kernel_type': CONVENIENT_TYPE, 'degree': 3, 'gamma': 1e-20, 'coef0': 1e-3}
    project, projection_matrix = kernel_LDA(X_train, y_train, args.embedding_dim, kernel_setting)
    print(projection_matrix.shape)
    classify_subject_kernel_pca(X_train, y_train, X_test, y_test, projection_matrix, args.k, kernel_setting)

    # PCA
    """
    CONVENIENT_TYPE = 'covariance'
    # CONVENIENT_TYPE = 'RBF'
    kernel_setting = {'kernel_type': CONVENIENT_TYPE, 'degree': 3, 'gamma': 1e-9, 'coef0': 1e-2}
    print(img_shape)
    print('-'*30)
    projection_matrix, eigen_faces = PCA(X_train, kernel_setting, img_shape, args.embedding_dim)
    print('-'*30)

    if SHOW_FACES:
        num = eigen_faces.shape[0]
        for i in range(num):
            plt.subplot(5, 5, i+1)
            plt.imshow(scale_up_img(eigen_faces[i], cv2.INTER_CUBIC, args.scale_down_times, args.scale_down_loop), cmap='gray')
            plt.axis('off')
        plt.show()


    # projection_matrix = np.load('PCA_W.npy')
    print('main')
    print('projection matrix', projection_matrix.shape)
    if kernel_setting['kernel_type'] == 'covariance':
        if SHOW_RECONS_FACES:
            reconstructed_faces = X_train.dot(projection_matrix.dot(projection_matrix.T)).reshape(-1, *img_shape)
            print('reconstructed_faces', reconstructed_faces.shape)
            shown_recons_faces = reconstructed_faces[np.random.randint(len(reconstructed_faces), size=10)]
            for i in range(5):
                plt.subplot(1, 5, i+1)
                plt.imshow(scale_up_img(shown_recons_faces[i], cv2.INTER_CUBIC, args.scale_down_times, args.scale_down_loop), cmap='gray')
                plt.axis('off')
            plt.show()
            print('shown_recons_faces', shown_recons_faces.shape)
        print('-'*30)
        classify_subject_normal(X_train, y_train, X_test, y_test, projection_matrix, args.k)
    else:
        classify_subject_kernel_pca(X_train, y_train, X_test, y_test, projection_matrix, args.k, kernel_setting)
    """
