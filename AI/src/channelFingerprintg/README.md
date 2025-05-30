# Key Generation
This module presents a key generation method that harnesses AI to mitigate the discrepancy between the quantized binary keys of the communicating nodes. 

![image](https://user-images.githubusercontent.com/46358777/222784708-e16af166-1f1f-4e1e-9686-039ee3da8dea.png)

Our experimental setting consisted of 8 Adalm-Pluto software-defined radios where 4 of these devices were used for collecting training and testing samples while the remaining 3 devices not included in the training stage were used for testing purposes only. The 7 Adalm-Pluto software-defined radios were controlled by 7 single-board computers that acted as the nodes in a single network. The single board computers consisted of 8 Nvidia Jetson Nano. Each of these nodes where controlled from a central control station that used the deployed API to control individual nodes and deploy different RF communication behaviors between the SDRs. The central control station also helped collect samples from devices.

The 7 Adalm-Pluto SDRs carrier frequency is 2.485 GHz, the transmission interval was set to 0.3s, the receiver sampling rate was set to 4 MHz, and the transmitted signal was a 1 MHz sinusoidal wave. The received packet for every transmission consisted of 8192 IQ samples.

We collected training samples using devices 1 through 4. For every round of data collection, it was chosen a set of 3 devices from the 4 available devices where two devices acted as Alice and Bob or the honest nodes, and a third device acted as Eve or the rogue node. Nodes Alice and Bob act as both transmitter and receiver since they exchange probes continuously while Eve only acts as receiver simulating an eavesdropper. The collected samples in both directions of Alice and Bob are regarded as using channel AB and BA while the data collected by Eve from Alice and Bob are regarded as using channels AC and BC respectively. In every scenario, the number of probes exchanged between Alice and Bob where 200, and Eve receives all 200 probes for a total of 400 packets generated from every scenario. In total, all 12 possible scenarios from the 4 devices were considered for collecting training data adding up to 4800 packets.

![image](https://user-images.githubusercontent.com/46358777/222783525-7d63a36b-037b-4bfe-bf94-c200b84a2642.png)
![image](https://user-images.githubusercontent.com/46358777/222783500-af24842d-eb2e-44c1-8e48-7cf8a66733b8.png)

We adopted deep metric learning which is used for training deep neural networks that return one-dimensional N-feature vectors.

In order to extract the channel characteristics of a signal it is necessary to determine the best input for the deep learning model to maximize its efficiency. Experimentally by extracting the time-frequency domain of our signals, it can be observed that the transmissions exchanged between Alice and Bob are relatively more similar than the transmissions obtained by Eve from both Alice and Bob. A signal is best analyzed by applying the short-time Fourier transform (STFT) which reveals the time-frequency characteristics of a signal. The input to the neural network is the spectrogram in dB scale which has dimensions TxM where T is the number of samples per segment and M is the number of frames.

![image](https://user-images.githubusercontent.com/46358777/222553209-aaa50a81-e9c8-4010-b5a2-85d2d569d482.png)

he neural network architecture follows the design proposed by \cite{GShen} with reference to the ResNet architecture which has been successfully adopted for feature extraction and has been optimized to be more lightweight. We have added an additional sigmoid activation layer as the output for quantization purposes such that the model optimizes based on a vector of values between 0 and 1. 

The input to the channel feature extractor is the two dimensional vector after applying a short-time Fourier transform to a signal. The neural network consists of one 32 7x7 filters with a stride of two convolutional layers, four 32 3x3 filters convolutional layers, four 64 3x3 filters convolutional layers, one average pooling layer, one 512-neuron dense layer, and one L2 normalization layer. Additionally, we included a sigmoid activation layer layer for returning an array of values ranging between 0 and 1.

![image](https://user-images.githubusercontent.com/46358777/222554294-77030316-35f6-4358-b581-4ccdece2c3d7.png)

The loss function in a neural network determines how well a model performs as it is training in order to tune the model's weights. The triplet loss objective is to minimize the euclidean distance between the anchor and the positive feature vectors while maximizing the distance between the anchor and the negative feature vectors. The anchor and positive are features extracted from channel alice-bob or channel bob-alice and the negative are features extracted from channel alive-eve or bob-eve.

![image](https://user-images.githubusercontent.com/46358777/222555118-9de1cf33-ef20-4000-8d52-455d220ce659.png)

In order to measure the performance of our proposed key generation method, four main metrics were considered. These are bit disagreement rate, key reconciliation rate, randomness, and key generation rate. During all assessments, 2 scenarios were considered. The first scenario consisted of devices 1, 2, and 3 which were included in the training of the channel feature extractor. The second scenario consisted of devices 5, 6, and 7 which were not included in the training of the channel feature extractor. Also for all test scenarios the number of probes exchanged by nodes A and B is 100.

Bit disagreement rate (BDR) consists of measuring the difference in bits between the keys generated by two sets of nodes. Successful key generation methods require the key disagreement rate to be smaller than the correction capacity of the implemented reconciliation algorithm.

During the first scenario, the BDR between the generated 512 binary vectors using the channel feature extractor averages 3.60\% BDR with its highest peak at 48\% BDR and the lowest value at 0\% BDR
![image](https://user-images.githubusercontent.com/46358777/222785215-1233de98-0194-4fb7-8b36-13d4129bf7eb.png)

In the second scenario, the BDR between the generated 512-binary vector using the channel feature extractor averages 2.43\% BDR with its highest peak at 67.19\% BDR and the lowest value at 0\% BDR
![image](https://user-images.githubusercontent.com/46358777/222785236-36df479b-ce47-44ea-832b-3a251e6e7235.png)

Given the generated keys Ya and Yb using the proposed AI-assisted method, then it is necessary to reconcile since there exist differences between the generated keys. The goal of the reconciliation method is to successfully reconcile the generated keys while keeping the amount of leaked information at its minimum. Our implementation uses the reed-salomon code for key reconciliation which is denoted by RS(N, K). 

 In order to measure the performance of implementing reed-salomon code for key reconciliation, we have measured the success rate for reconciliation for 100 keys generated by Alice and Bob on 3 different settings for RS(N,K). 
 
 During the first scenario considered for RS(255,128), RS(191,128), and RS(159,128) the average Alice-Bob reconciliation rate obtained was 0.96% , 0.96%, and 0.9% respectively.
 
 ![image](https://user-images.githubusercontent.com/46358777/222785685-2709d543-4a44-48d8-b99b-42d9e25d3317.png)
 
 During the second scenario considered for RS(255,128), RS(191,128), and RS(159,128) the average Alice-Bob reconciliation rate obtained was 0.97% , 0.96%, and 0.86% respectively.
 
 ![image](https://user-images.githubusercontent.com/46358777/222785724-d52936fe-0d8a-4f9d-a3fa-07541b10ad2d.png)

Randomness in key generation methods has certain requirements for which the National Institute of Standards and Technology has created a statistical randomness test suite. The NIST test suit consists of 15 tests to evaluate different randomness features. During this research, it was considered the Monobit, Frequency Within Block, Runs, Longest Run Ones in A Block, Discrete Fourier Transform, Non-Overlapping Template Matching, Serial, Approximate Entropy, Cumulative Sums, Random Excursion, and Random Excursion Variant.

![image](https://user-images.githubusercontent.com/46358777/222786488-753c94c3-8966-4d63-84f0-f3e0e84115a9.png)
![image](https://user-images.githubusercontent.com/46358777/222786096-88644ff8-dc7b-47ad-a330-097670cb97fb.png)

