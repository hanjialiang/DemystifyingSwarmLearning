import tensorflow.keras as K
import pandas as pd
import argparse
import pickle


def test_model_on_generator(model, test_generator, verbose):
    result = model.evaluate(test_generator, verbose=verbose)
    return {
        'metrics': model.metrics_names,
        'result': result,
    }

def get_test_generator(test_path, data_csv, image_dir, input_shape, batch_size):
    df = pd.read_csv(data_csv, index_col='Image Index', converters={'Finding Labels': lambda x: [i for i in x.split('|') if i != 'No Finding']}).drop(
        ['Follow-up #', 'Patient ID', 'Patient Age',
        'Patient Gender', 'View Position', 'OriginalImage[Width', 'Height]',
        'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11'], axis=1
    )
    df.head()

    all_labels = list(sorted(set(y
                for x in df['Finding Labels']
                for y in x
                )))

    with open(test_path, 'r') as fp:
        idx = [x.strip() for x in fp.readlines()]
    test_df = df.loc[idx].reset_index()
    test_df.head()


    datagen = K.preprocessing.image.ImageDataGenerator(
        height_shift_range=5/100,
        width_shift_range=5/100,
        rotation_range=5,
        zoom_range=15/100,
        rescale=1./255,
        samplewise_center=True,
        samplewise_std_normalization=True,
    )

    test_generator = datagen.flow_from_dataframe(
        test_df,
        image_dir,
        x_col='Image Index',
        y_col='Finding Labels',
        target_size=input_shape,
        color_mode='rgb',
        classes=all_labels,
        batch_size=batch_size
    )

    return test_generator

def test_model_on_data(model_path, test_path, data_csv, image_dir, input_shape, batch_size, verbose=False):
    model = K.models.load_model(model_path)
    test_generator = get_test_generator(test_path, data_csv, image_dir, input_shape, batch_size)
    return test_model_on_generator(model, test_generator, verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use", required=True)
    parser.add_argument("--data", type=str, help="Test data to use", required=True)
    parser.add_argument("--name", type=str, help="Evaluation name", required=True)
    args = parser.parse_args()

    # model_path = 'saved/SplitResult-gender/LL-1/NIHXray-DenseNet49-BS32-LR0.001-random-notfreeze-CL-20210825030549-TRAIN_N26058-Adam-27'
    # test_path = 'SplitResult-age/1/test_list.txt'
    name = args.name
    model_path = args.model
    test_path = args.data
    data_csv = '/Users/maghsk/Gits/MergeWeights/NIH_Chest_X-rays/Data_Entry_2017.csv'
    image_dir = '/Volumes/HYD-DATA/NIH_Chest_X-rays/images'
    input_shape = (256, 256)
    batch_size = 32
    res = test_model_on_data(model_path, test_path, data_csv, image_dir, input_shape, batch_size)
    with open(f'{name}-result.pickle', 'wb') as fp:
        pickle.dump(res, fp)
