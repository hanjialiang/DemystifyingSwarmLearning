import pickle
from re import VERBOSE
from test_model_on import test_model_on_generator, get_test_generator
import IPython
import gc
import tqdm
import tensorflow.keras as K

if __name__ == '__main__':
    model_list = [
        ('LL-1', 'saved/SplitResult-gender/LL-1/NIHXray-DenseNet49-BS32-LR0.001-random-notfreeze-CL-20210825030549-TRAIN_N26058-Adam-27'),
        ('LL-2', 'saved/SplitResult-gender/LL-2/NIHXray-DenseNet49-BS32-LR0.001-random-notfreeze-CL-20210825025912-TRAIN_N28507-Adam-22'),
        ('LL-3', 'saved/SplitResult-gender/LL-3/NIHXray-DenseNet49-BS32-LR0.001-random-notfreeze-CL-20210825025916-TRAIN_N31947-Adam-21'),
        ('SL-1', 'saved/SplitResult-gender/SL-1/NIHXray-DenseNet49-BS32-LR0.001-random-notfreeze-SL_1_1_W50_SINT900-20210825090055-TRAIN_N26058-Adam-50'),
        ('SL-2', 'saved/SplitResult-gender/SL-2/NIHXray-DenseNet49-BS32-LR0.001-random-notfreeze-SL_1_2_W50_SINT900-20210825090056-TRAIN_N28507-Adam-46'),
        ('SL-3', 'saved/SplitResult-gender/SL-3/NIHXray-DenseNet49-BS32-LR0.001-random-notfreeze-SL_1_3_W50_SINT900-20210825090058-TRAIN_N31947-Adam-42'),
    ]

    data_list = [
        ('1', 'SplitResult-gender/1/test_list.txt'),
        ('2', 'SplitResult-gender/2/test_list.txt'),
        ('3', 'SplitResult-gender/3/test_list.txt'),
    ]


    # data_csv = '/Users/maghsk/Gits/MergeWeights/NIH_Chest_X-rays/Data_Entry_2017.csv'
    # image_dir = '/Volumes/HYD-DATA/NIH_Chest_X-rays/images'
    data_csv = '/home/yudonghan/storage/NIHCHEST/NIH_Chest_X-rays/Data_Entry_2017.csv'
    image_dir = '/home/yudonghan/storage/NIHCHEST/NIH_Chest_X-rays/images'
    input_shape = (256, 256)
    batch_size = 128

    gen_list = [ (name, get_test_generator(test_path, data_csv, image_dir, input_shape, batch_size)) for name, test_path in data_list ]

    results = []
    for model_name, model_path in model_list:
        K.backend.clear_session()
        model = K.models.load_model(model_path)
        for gen_name, generator in gen_list:
            print('Testing {} on {}'.format(model_name, gen_name))
            name = f'{model_name}_on_DS-{gen_name}'
            res = test_model_on_generator(model, generator, verbose=True)
            results.append((name, res))
            print(res)

        del model
        gc.collect()


    try:    
        results = { name: res for name, res in results }
        with open('results.pkl', 'wb') as f:
            pickle.dump(results, f)
    except:
        IPython.embed()
