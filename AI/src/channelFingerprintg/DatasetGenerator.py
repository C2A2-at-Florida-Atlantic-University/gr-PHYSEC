import h5py
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
import huggingface_hub as hf
import datasets

class DatasetGenerator():
    def __init__(self, fileName):
        self.fileName = fileName
        self.dataFrame = self.generate_dataframe_from_hdf5()        

    # Save dataframe to huggingface dataset
    def saveDataFrame(self, dataset_name, config_name, repo_name="CAAI-FAU"):
        username = hf.whoami()['name']
        print("Pushing dataset to Hugging Face hub...")
        # Create a new dataset
        print("Creating new dataset...")
        # Convert to HF Dataset
        hf_dataset = Dataset.from_pandas(self.dataFrame)
        # Push to Hugging Face hub
        repo_name = username if repo_name == "" else repo_name
        hf_dataset.push_to_hub(
            repo_name+"/"+dataset_name, 
            private=True, 
            config_name=config_name
        )
        print("Dataset pushed to Hugging Face hub successfully.")

    # read hdf5 file
    def read_hdf5_file(self):
        data = {}
        with h5py.File(self.fileName, 'r') as f:
            # Assuming the dataset is named 'dataset'
            # Read all key from the file
            keys = list(f.keys())
            print("Keys in HDF5 file: ", keys)
            for key in keys:
                data[key] = f[key][:]
        return data

    def generate_dataframe_from_hdf5(self):
        data = self.read_hdf5_file()
        for key in data.keys(): # Read all keys in dictionary
            if data[key].shape[0] == 1: # If first dimension shape is 1, remove it
                data[key] = data[key][0]
            if len(data[key].shape) > 1: # If data shape is 2D, make it a list
                data[key] = list(data[key])
        return pd.DataFrame(data)

    def separate_iq_samples(self, data):
        # I is the second half of the samples
        I = data[len(data)//2:]
        # Q is the first half of the samples
        Q = data[:len(data)//2]
        return I, Q

    def plot_iq_samples(self, I,Q):
        # Plot IQ samples
        plt.plot(I, label='I')
        plt.plot(Q, label='Q')
        plt.title('IQ Samples')
        plt.xlabel('Sample Number')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    def plot_quadruplet_samples(self, id):
        # Plot quadrupole samples
        dataInstance = self.dataFrame[self.dataFrame['ids'] == id]
        for index, row in dataInstance.iterrows():
            print("Index: ", index)
            I = row["I"]
            Q = row["Q"]
            plt.plot(I, label='I')
            plt.plot(Q, label='Q')
            name = "Eve" if row["instance"] % 2 == 0 else "Alice" if row["instance"] == 1 else "Bob"
            plt.title(f'IQ Samples. ID: {id}, label: {row["channel"]}, instance: {row["instance"]} for {name}')
            plt.xlabel('Sample Number')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()
    
def load_OTA_lab_config():
    nodeConfigs = {
        "IDs":[
            # [1,2,3],
            # [1,4,5],
            # [1,4,8],
            # [2,4,3],
            # [4,2,5],
            # [4,2,8],
            # [4,8,5],
            [5,7,8],
            [5,8,7],
            [8,4,1],
            [8,5,1],
            [8,5,4]
        ],
        "Timestamps":[
            # 1747672681,
            # 1747678946,
            # 1747679768,
            # 1747673451,
            # 1747675004,
            # 1747674226,
            # 1747716767,
            1747697115,
            1747699032,
            1747713786,
            1747709125,
            1747706583,
        ]
    }
    return nodeConfigs

def load_OTA_dense_config():
    nodeConfigs = {
        "IDs":[
            [1,2,3],
            [1,2,5],
            [1,3,2],
            [4,3,5]
        ],
        "Timestamps":[
            1747695341,
            1747776498,
            1747698126,
            1747770576
        ]
    }
    return nodeConfigs

if __name__ == "__main__":
    # Example usage
    saveDataFrame = True
    names = ['Alice', 'Bob', 'Eve']
    
    site = "Powder-OTA-Lab" # "Powder-OTA-Lab" or "Powder-OTA-Dense"
    if site == "Powder-OTA-Lab":
        nodeConfigs = load_OTA_lab_config()
    else:
        nodeConfigs = load_OTA_dense_config()
    
    numProbes = 100
    signalType = "sinusoid"
    folder = "/Users/josea/Workspaces/PowderKeyGen/"
    for nodeIDs, timestamp in zip(nodeConfigs["IDs"], nodeConfigs["Timestamps"]):
        file = folder + "Dataset_"+("" if site == "Powder-OTA-Lab" else "OTADense_")+"Channels_"+signalType+"_"+str(numProbes)+"_"+"".join(str(node) for node in nodeIDs)+"_"+str(timestamp)+".hdf5"
        print("Reading file: ", file)
        # Read the hdf5 file
        dataset = DatasetGenerator(file)
        # Print the dataframe info
        dataset.dataFrame.info()
        # print("Dataframe:", dataset.dataFrame)
        # Plot the IQ samples
        # dataset.plot_iq_samples(0, 1)
        # Plot the quadruplet samples
        # dataset.plot_quadruplet_samples(3)
        if saveDataFrame:
            dataset_name = "Key-Generation"
            config_name = "Sinusoid-"+site+"-Nodes-"+"".join(str(node) for node in nodeIDs)  #"Sinusoid-Powder-OTA-Lab" 
            repo_name="CAAI-FAU"
            dataset.saveDataFrame(dataset_name, config_name, repo_name)
            print("Dataframe saved to huggingface.")
            # Load the dataset from huggingface
            dataset = datasets.load_dataset(repo_name+"/"+dataset_name, config_name)
            # Dataset information
            print("Dataset information: ", dataset)