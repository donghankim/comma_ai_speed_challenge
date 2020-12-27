import argparse

def get_args():
    argp = argparse.ArgumentParser(description='speed challenge',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # GENERAL
    argp.add_argument('--device', type=str, default='cpu')
    argp.add_argument('--train', action = 'store_true')
    argp.add_argument('--val', action = 'store_true')
    argp.add_argument('--test', action = 'store_true')

    # PATH
    argp.add_argument('--root', type=str, default="/Users/donghankim/Desktop/Coding/public_github/comma_ai_speed_challenge/data/")
    argp.add_argument('--train_vid', type = str, default="/Users/donghankim/Desktop/Coding/public_github/comma_ai_speed_challenge/video/train.mp4",)
    argp.add_argument('--test_vid', type = str, default = "/Users/donghankim/Desktop/Coding/public_github/comma_ai_speed_challenge/video/test.mp4")
    argp.add_argument('--train_frames', type = str, default = "/Users/donghankim/Desktop/Coding/public_github/comma_ai_speed_challenge/data/train/")
    argp.add_argument('--test_frames', type = str, default = "/Users/donghankim/Desktop/Coding/public_github/comma_ai_speed_challenge/data/test/")
    argp.add_argument('--train_speeds', type = str, default = "/Users/donghankim/Desktop/Coding/public_github/comma_ai_speed_challenge/data/train_speed.txt")
    argp.add_argument('--csv_path', type = str, default = "/Users/donghankim/Desktop/Coding/public_github/comma_ai_speed_challenge/data/labels.csv")
    argp.add_argument('--model_path', type=str, default="/Users/donghankim/Desktop/Coding/public_github/comma_ai_speed_challenge/model_weights/")
    argp.add_argument('--result_path', type=str, default="/Users/donghankim/Desktop/Coding/public_github/comma_ai_speed_challenge/results/")

    # PREPROCESS
    argp.add_argument('--preprocess_frames', action = 'store_true')
    argp.add_argument('--create_csv', action = 'store_true')

    # MODEL SELECTION
    argp.add_argument('--model', type = str, default = "simpleCNN", choices = ['simpleCNN', 'Resnet'])

    # TRAINING PARAMETERS
    argp.add_argument('--epochs', type = int, default = 5)
    argp.add_argument('--lr', type = float, default = 0.0001)
    argp.add_argument('--batch_size', type = int, default = 100)

    return argp.parse_args()


