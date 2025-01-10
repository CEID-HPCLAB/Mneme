import os
from sklearn.model_selection import train_test_split
from _make_classification import make_classification
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", help = "Number of samples", type = int, default = 50000, required = False)  
    parser.add_argument("--features", help = "Number of features", type = int, default = 5, required = False) 
    parser.add_argument("--test_size", help = "test size", default = 0.2, type = float, required = False)
    parser.add_argument("--chunks", help = "chunk size", default = 10, type = int, required = False)
    parser.add_argument("--classes", help = "Number of classes", default = 2, type = int, required = False)
    parser.add_argument("--path", help = "New Data File's path", default = "./", type = str, required = False)
    parser.add_argument("--problem_name", "-pbn", help = "problem name", default = "xyz_40K", type = str, required = False) 

    args = parser.parse_args()
    
    # 2M n_samples x 7k n_features -> ~100GB csv output size | 2M n_samples x 700 n_features -> ~10GB csv output size
    n_samples = args.samples
    n_features = args.features 
    test_size = args.test_size
    n_chunks = args.chunks
    path = args.path
    problemname = args.problem_name
    n_classes = args.classes
    
    train_file = os.path.join(path, f'train_data_{problemname}.csv')
    test_file = os.path.join(path, f'test_data_{problemname}.csv')

    if os.path.exists(train_file):
        os.remove(train_file)

    if os.path.exists(test_file):
        os.remove(test_file)

    use_index = False   
    reset_index = False 

    chunk_size = n_samples // n_chunks  

    for i in range(n_chunks):
        X, y = make_classification(n_samples = chunk_size, n_features=n_features, n_informative=n_features, n_redundant=0,
                                n_clusters_per_class=1, flip_y=0.01,
                                n_classes=n_classes, random_state=42+i,
                                fixed_random_state=24,  
                                shuffle=True,
                                shuffle_features=False) 
        y = y.reshape(-1, 1)
        df1 = pd.DataFrame(data=X)
        df2 = pd.DataFrame(data=y)
        df = pd.concat([df1, df2], axis=1)

        if i == 0:
            use_header = True
        else:
            use_header = False
        df.columns =[f"x{i}" for i in range(n_features)] + [f"y{i}" for i in range(y.shape[1])]

        # if use_index:
        #     length = len(df.index)
        #     start = i*length
        #     end = start + length
        #     df.index = list(range(start, end))

        train, test = train_test_split(df, test_size=test_size)
        # if use_index and reset_index:
        #     length = len(train.index)
        #     start = i*length
        #     end = start + length
        #     train.index = list(range(start, end))

        #     length = len(test.index)
        #     start = i*length
        #     end = start + length
        #     test.index = list(range(start, end))

        # if True, we save the partial files
        # if False:
        #     train_file_part = path + '/train_data_{}_{:04d}.csv'.format(problemname, i)
        #     test_file_part = path + '/test_data_{}_{:04d}.csv'.format(problemname, i)

        #     if os.path.exists(train_file_part):
        #         os.remove(train_file_part)

        #     if os.path.exists(test_file_part):
        #         os.remove(test_file_part)

        #     train.to_csv(train_file_part, header=True, index=use_index, float_format="%.6f")
        #     test.to_csv(test_file_part, header=True, index=use_index, float_format="%.6f")

        train.to_csv(train_file, mode="a", header=use_header, index=use_index, float_format="%.6f")
        test.to_csv(test_file, mode="a", header=use_header, index=use_index, float_format="%.6f")