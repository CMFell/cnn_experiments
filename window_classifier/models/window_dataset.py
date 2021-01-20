from torch.utils.data import Dataset
from torchvision import transforms

class WindowTestDataset(Dataset):

    def __init__(self, windows_list):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.windows_list = windows_list
        self.transform = transforms.Compose([Resize((299, 299)),
                                             ToTensor(),
                                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.windows_list)

    def __getitem__(self, idx):
        image = self.windows_list[idx]
        image = self.transform(image)

        return image