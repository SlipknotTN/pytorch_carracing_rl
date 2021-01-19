import configparser


class ConfigParams(object):

    def __init__(self, file):

        config = configparser.ConfigParser()
        config.read_file(open(file))

        # Model
        self.input_num_frames = config.getint("MODEL", "input_num_frames")

        # Train
        self.num_episodes = config.getint("TRAIN", "num_episodes")
        self.initial_epsilon = config.getfloat("TRAIN", "initial_epsilon")
