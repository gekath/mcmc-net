import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from rbm import *


def load_netflix(dir_name, N, M, K):

    user_ids = {}
    user_ratings = np.zeros((N, M))

    for i in range(M):

        filename = 'mv_%07d.txt' % (i + 1)
        f = open(os.path.join(dir_name, filename), 'r')
        data = f.readlines()[1:]
        f.close()

        user_count = 0
        num_ratings = len(data)
        for j in range(num_ratings):
            user, rating, date = data[j].split(',')
            if user in user_ids:
                user_idx = user_ids[user]
                user_ratings[user_idx, i] = int(rating)
            else:
                if user_count < N:
                    user_ids[user] = user_count
                    user_ratings[user_count, i] = int(rating)
                    user_count += 1
                else:
                    break

        print(i, num_ratings)

    return user_ratings, user_ids


def write_to_file(ratings, user_id_dict):

    user_id_filename = 'netflix/training_set/user_id_dict.txt'
    ratings_filename = 'netflix/training_set/ratings.txt'

    np.savetxt(ratings_filename, ratings)

    f = csv.writer(open(user_id_filename, 'w'))
    for key, val in user_id_dict.items():
        f.writerow([key, val])


if __name__ == "__main__":

    directory = '../data/netflix'
    # num_users = 480189
    # num_movies = 17770
    num_users = 2000
    num_movies = 1000
    max_rating = 5
    num_epochs = 50
    batchsize = 100
    temps = np.arange(0.9, 1, 0.025)
    rmse_array = []

    print('Beginning to load netflix data')
    user_ratings, user_id_dict = load_netflix(directory, num_users, num_movies, max_rating)
    # write_to_file(user_ratings, user_id_dict)
    print('Finished loading netflix data')

    print('Initializing RBM')
    rbm = RBM(num_movies, 1000)
    rbm.params[:] = np.random.uniform(-1./10, 1./10, len(rbm.params))
    print('Training RBM')
    trainer = CDTrainer(rbm)

    for i in range(len(temps)):
        rmse = trainer.train(user_ratings, num_epochs, batchsize=batchsize,
                             temp=temps[i], sampler='Gibbs')
        rmse_array.append(rmse)

    epochs = np.arange(num_epochs)

    plt.figure(1)
    plt.plot(epochs, rmse_array[0], 'ro', epochs, rmse_array[1], 'bs', epochs,
             rmse_array[2], 'g^', epochs, rmse_array[3], 'r--')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.savefig('RMSE_tempered_gibbs.png')
    plt.show()