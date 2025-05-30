import datetime as dt
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str
import os 

#Dataset Generator for creating datasets
#Creates IQ datasets within the SigMF format
#IQ files as well as signal metadata
class sigMFDataset():
    def __init__(self):
        self.date_time = dt.datetime.utcnow().isoformat()+'Z'
        self.metadataIsSet = False
        
    def setData(self,data,label,samplesPerExample):
        self.data = data
        self.label = label
        self.SPE = samplesPerExample
        
    def createDataset(self):
        if self.metadataIsSet:
            self.createFolder()
            self.createIQFile()
            self.createMetadata()
        else:
            print("Set metadata first with setMetadata()")
    
    def createFolder(self):
        parent_dir = os.getcwd()
        directory = self.fileName+"_"+self.author+"_"+self.date_time
        self.path = os.path.join(parent_dir,directory)
        os.mkdir(self.path) 
        print("Directory '% s' created" % directory) 
    
    def createIQFile(self):
        self.data.tofile(self.fileName+'.sigmf-data')
    
    def setMetadata(self):
        # create the metadata
        self.fileName = input("File Name:")
        self.samp_rate = input("Sampling Rate:")
        self.freq = input("Sampling Frequency:")
        self.author = input("Author Email:")
        self.description = input("Description:")
        self.metadataIsSet = True
        
    def setMetadata(self,fileName,samp_rate,freq,author,description):
        # create the metadata
        self.fileName = fileName
        self.samp_rate = samp_rate
        self.freq = freq
        self.author = author
        self.description = description
        self.metadataIsSet = True
    
    def createMetadata(self):
        self.metadata = SigMFFile(
            data_file=self.fileName+'.sigmf-data', # extension is optional
            global_info = {
                SigMFFile.DATATYPE_KEY: get_data_type_str(self.data),  # in this case, 'cf32_le'
                SigMFFile.SAMPLE_RATE_KEY: self.samp_rate,
                SigMFFile.AUTHOR_KEY: self.author,
                SigMFFile.DESCRIPTION_KEY: self.description,
                SigMFFile.FREQUENCY_KEY: self.freq,
                SigMFFile.DATETIME_KEY: self.date_time,
            }
        )
        self.metadata.tofile(self.fileName+'.sigmf-meta')

def testDatasetGenerator():
    return True

if __name__ == '__main__':
    testDatasetGenerator()
    print("Done")