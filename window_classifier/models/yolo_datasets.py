from torch.utils.data import Dataset
from torchvision import transforms

class TileImageTestDataset(Dataset):

    def __init__(self, tiles_list):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.tiles_list = tiles_list
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.tiles_list)

    def __getitem__(self, idx):
        image = self.tiles_list[idx]
        image = self.transform(image)

        return image