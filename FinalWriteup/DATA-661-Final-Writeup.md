
# Implementing Convolutional Neural Networks for Image Classification and Facial Recognition Using Tensorflow v1.0: An Independent Study

#### *By James Topor, Spring 2017*

## Problem Formulation & Objectives

Facial recognition technology is becoming a widely used tool for the identification of individuals for many different purposes, including law enforcement, security and social media auto-tagging of images. Neural networks have played a crucial role in enabling the practical application of facial recognition technology and will continue to do so as interest in the use of facial recognition technology expands. In particular, convolutional neural networks ('CNN's') have been shown to be effective when applied to image classification and facial recognition tasks. However, there appears to be little consensus regarding how the architecture of a CNN should be configured for optimal performance. In fact, although researchers have investigated CNN's for many years now, the exact reasons why CNN's behave the way they do are still not fully understood, and  they remain a somewhat opaque and highly empirical field of study.

This context provides an opportunity for those who are new to the study of neural networks to not only learn the basic mathematical precepts that underlie neural network theories, but to also explore the application of such theories via an empirical approach. The purpose of this independent study was to apply Google's TensorFlow API to the design, implementation and evaluation of image classification and facial recognition systems using convolutional neural network architectures.  The goals of the study were as follows:

- To allow the author to develop a thorough understanding of the concepts that underlie CNN theory;

- To gain experience with the implementation of CNN's using Tensorflow v 1.0;

- To gain insight as to how well CNN's implemented via Tensorflow perform when applied to image classification and facial recognition problems;

- To use Tensorflow for purposes of implementing "transfer learning", wherein components of a CNN trained for one task are re-purposed for another task without the need for re-training of the components extracted from the original CNN.


## Literature Review

A necessary prerequisite for the achievement of these goals was a review of the wide variety of relevant neural networking theory educational materials and journal articles. Stanford University's CS231n materials [1] proved to be particularly useful for purposes of acquiring knowledge about how to design and implement neural networks for image classification tasks. Their tutorials provide a clear explanation of the core concepts of neural network theory, including optimization algorithms(e.g., stochastic gradient descent; AdaGrad, Adam, etc..), backpropagation, regularization, CNN layer patterns, MaxPooling, CNN computational requirements, and loss functions. Tensorflow's website [1] also proved to be a great point of reference for general machine learning knowledge as well as for specific examples of how to implement various image classification algorithms [3, 4, 5].

While the references indicated above can provide the reader with a solid foundation in both neural networks in general and CNN's in particular, additional materials specific to the implementation of CNN's for facial recognition tasks were also identified. Yi et al. [6] not only generate a large scale (and now publicly available) set of nearly 500,000 facial images, but also implement an 11-layer CNN using ReLU neurons and small filters to achieve facial recognition accuracy exceeding that of FaceBook's "DeepFace" system. Of particular interest for this study was the approach they use for constructing a high-performance CNN for facial recognition. While their 11-layer CNN proved to be too demanding for our available hardware and software, their general approach to filter sizing, loss functions, and pooling served as a key point of reference for the CNN's discussed herein.

Garcia and Delakis [7] provide the basis for most research into the use of CNN's for facial recognition tasks. Their 2004 IEEE paper applies a 7-layer CNN which at the time outperformed most other facial recognition algorithms. As such, their work provides the researcher with a foundational level of background and insight as to why CNN's are effective for facial recognition.  Finally, Steffan Duffner's 2007 Dissertation [8] objectively evaluates Garcia and Delakis's approach while providing additional relevant background and insight as to why CNN's are effective for facial recognition. 


## Approach

Knowledge gained during the literature view process was then used to construct multiple CNN models for a variety of image classification and facial recognition tasks. The performance of the various CNN architectures used were compared against one another via performance metrics such as loss/cost and accuracy. The CNN's were implemented using __Python__, __iPython Notebook__, Google's __Tensorflow__ API, and the following publicly available data sets:

- __MNIST__: A database of 70,000 28x28 monochrome images of handwritten digits (0, 1, ..., 9) [9];

- __CIFAR-10__: A database of 60,000 32x32 color images belonging to 10 possible classifications (6,000 images per class) [10];

- __CIFAR-100__: A database of 60,000 32x32 color images belonging to 100 possible classifications (600 images per class) [11];

- __Labeled Faces in the Wild__ ('LFW'): A database of 13,233 250x250 color facial images of 5,749 different people [12];

- __CASIA WebFace__: A database of 494,414 250x250 color and monochrome facial images of 10,575 different people [13];

The specifics of the CNN architectures used for each of these data sets necessarily varied due to the heterogeneity of the data sets. However, some general principles applied to each:

1. A "deeper" CNN is more effective than a "wider" CNN. Empirical work cited in the references provided herein indicates that configuring a CNN with a small number of layers that each contain many neurons will tend to underperform a CNN configured with additional layers that are each comprised of fewer neurons.

2. Relatively small input batch sizes were used to train each model (e.g., 64 to 128 images). Use of smaller batch sizes allows relatively large data sets to be used for model training while also decreasing the likelihood that the resulting CNN model will be overfit relative to the training data.

3. The dimensionality of the images used as input to the CNN should be highly divisible by 2 (if possible, a power of 2). For example, images of size 32x32, 64x64, 128x128, etc. are preferable to images of size 71x37 or 111x81. This guideline serves to enhance the performance of the CNN. 

4. The conolutional layers of the CNN should make use of relatively small filter sizes and a stride of 1. Typically, filter sizes should not exceed 5x5, with 3x3 being preferable if a more granular leverl of feature extraction is required by the application.

5. Pooling layers should make us of "maxpooling", wherein the input data is downsampled to reduce its dimensionality. Each maxpooling layer should make use of 2x2 filters with a stride of 2. This guideline has the effect of discarding 75% of the input data: for example, if a 64x64 image is downsampled using a 2x2 filter with a stride of 2, the resulting output will be a 32x32 image, which, in fact, contains 75% less information than a 64x64 image.

6. Each fully connected layer is followed by a dropout layer to reduce the likelihood that the CNN will overfit its training data. Since the output of a fully connected layer will be substantially more dense than that of a pooling layer, a 50% to 60% retention ratio is applied within the dropout layer in such instances.

7. Where both necessary and computationally feasible, dropout layers are implemented following a pooling layer to reduce the likelihood that the CNN will overfit the training data. Since the output of a pooling layer will be much less dense than that of a fully connected layer, a 75% to 80% retention ratio is used in any dropout layer that follows a pooling layer so as to prevent underfitting of the model's training data.

8. A SoftMax function is used as the final layer of each CNN to calculate the probabilities of each input item relative to the model's possible classifications.

Construction of each CNN model was followed by model training and evaluation using dedicated training and testing subsets extracted from the model's data set. The efficacy of each training effort was evaluated relative to the output of the model's loss function. In general, each model was trained for an increasing number of steps until no substantive improvement in the loss function was observed. Evaluation of a CNN model is then measured by calculating the model's accuracy when applied to it's testing subset as well as the results of the loss function.


```python
%pylab inline
```

## The Models

The CNN model for each data set is discussed below. Please note that all python code either excerpted herein or accessible via the web links provided below is specific to Python 3.5.2, Tensorflow 1.0, and Windows 10. The author assumes no responsibility for your failure to properly install and/or configure these required items should you choose to make use of any of this code within your local environment.

### MNIST

As a first attempt at both constructing a functional neural network and using Tensorflow, the following Tensorflow tutorial proved invaluable:

https://www.tensorflow.org/tutorials/layers

The tutorial not only demonstates a straightforward approach to the implementation of a CNN in Tensorflow V1.0, it serves as as an introduction to Tensorflow's evolving __TF Layers__ API, which supposedly will be recommended as a preferred tool for the implemantation of CNN's via tensorflow going forward. After appropriately installing and configuring Tensorflow v1.0 and Python v3.5.2, the code shown in the tutorial was imported verbatim. One small change was required to get the tutorial code to function properly due to an error in the tutorial code which has since been corrected by Google on the tutorial's web page. As such, the code as used  now matches the code shown in the tutorial. The code used can be found at the following Github link:

https://github.com/jtopor/CUNY-MSDA-661/blob/master/MNIST-CNN/TF-Layers-CNN-MNIST.py

The MNIST data set is fairly simple in nature: the items contained within it are 28x28 monochrome images of hand drawn digits. An example of its content is shown below.

![Image](MNIST-Samp.png)

As we can see, the images have a white background devoid of detail. Therefore, a CNN designed to classify similar images requires a relatively low level of complexity due to the relative ease with which the digits can be found within the image and subsequently interpreted. The Tensorflow tutorial does not make use of the full MNIST data set. Instead, a total of 65,000 images are loaded by the tutorial code, with 55,000 used for training the CNN model and 10,000 used for evaluation. 

The structure of the CNN provided in the tutorial is shown below.

![Image](MNIST-Model.png)

As we can see above, the model is comprised of 2 pairings of convolutional / pooling layers followed by a flattening of the output of the second pooling layer. Each convolutional layer applies a 5x5 filter with stride 1 and each pooling layer applies a 2x2 MaxPool filter with stride 2. This conforms with the general guidelines outlined above in the __Approach__ section. The flattened output of the second pooling layer is fed to a fully connected dense layer containing 1024 neurons. The output of the fully connected layer is then passed through a dropout layer with a retention ratio of 60% (40% of the outputs are randomly disabled). Finally, a SoftMax function calculates the probabilities of any given image belonging to one of the 10 possible classifications (0 through 9).

The model is trained using a Stochastic Gradient Descent (SGD) optimizer and a learning rate of 0.001, as shown in the following code excerpt:



```python
# Configure the Training Op (for TRAIN mode)
if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")
```

After training the model for a total of 5,000 steps over the course of approximately one hour, we observed the output of the loss function as 0.2104. While it is possible the loss might have been decreased further as a result of continuing the training process for some unknown additinal quantity of steps, the training was terminated to allow for the evaluation of the model. Applying the evaluation data subset showed that the model achieved 92.09% accuracy with a loss of 0.2775. Since the MNIST model code is simply a nearly verbatim copy of the Tensorflow tutorial, no further training or testing was conducted with this model so as to allow for more time to be alloted to developing CNN models for the other data sets.

### CIFAR-10

The CIFAR-10 data set represents a huge leap in data complexity over the MNIST data set, with all images being color rather than monochrome and also containing signficantly more complicated subjects and backgrounds. A sample of images from the data set's 10 possible classifications is shown below.

![Image](CIFAR10-Samp.png)

The data set itself must be downloaded from the website indicated in [10]. Once downloaded, the user must decompress the resulting file and make its six component files (each containing 10,000 images) accessible to python. The user must then invoke python's __pickle.load()__ function to load the data set's pre-built python __dict__ objects from each of the six component files. That object must then be reshaped to match the required format of Tensorflow's tensor objects as follows:


```python
# assumes 'd' is the dict object obtained from the pickle.load() function
data = d['data']

# color image data MUST be converted to floating point format for Tensorflow
data = data.astype('float32')

# reshape the image data to match the required Tensor format
train_data = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color

# extract corresponding classification labels for images from dict object and load them into a numpy array
train_labels = np.asarray(d['labels'], dtype=np.int32)
```

All five training sub-batches should be combined into a single 4-dimensional array prior to any attempt to train a CIFAR-10 CNN model.

A first attempt at a CNN model for the CIFAR-10 data set built upon the MNIST sample code and yielded the following model structure:

![Image](CIFAR10-Mod1.png)

The source code for this model can be found at the following Github link:

https://github.com/jtopor/CUNY-MSDA-661/blob/master/CIFAR-CNN/TF-Layers-CIFAR-GITHUB.py

As we can see in the image above, this model varied from the MNIST model in that it made use of pairs of convolutional layers  stacked together prior to the application of a pooling layer. Furthermore, the second pair of convolutional layers rely on a 3x3 filter size in an attempt to extract more detail from the relatively complicated CIFAR-10 images than was required for the analysis of the MNIST images. After training the model for a total of 20,000 steps over the course of approximately seven hour, we observed the output of the loss function as 0.6919. Applying the evaluation data subset showed that the model achieved 63.5% accuracy with a loss of 1.036. 

A slight modification to this model was made to see whether increasing the number of filters used in the last convolutional layer would improve its performance. The optimizer was also changed from SGD to AdaGrad in an attempt to get the learning rate to automatically decay (SGD optimization not being capable of automatically degrading the learning rate). The architecture of the modified model is shown below.

![Image](CIFAR10-Mod2.png)

Training this model for 5,000 steps indicated that no significant improvement could be expected relative to the first CIFAR-10 model described above. Therefore, an attempt was made to identify other architectures that might prove more effective. Two  references in particular yielded useful insight [14, 15], with each author suggesting that an aggressive use of dropout layers after each pooling layer as well as the addition of a second fully connected layer might be appropriate. A revised model was generated and is shown below.

![Image](CIFAR10-Mod3.png)

The full source code for this model can be found at the following Github link:

https://github.com/jtopor/CUNY-MSDA-661/blob/master/CIFAR-CNN/TF-Layers-CIFAR-GITHUB-v3.py

As we can see in the image  above, the architecture of the model has been extemded by the addition of a total of 3 droput layers and one 256 element dense layer. Additionally, the size of the first dense layer has been reduced from 1024 elements to 512. Furthermore, the optimizer was switched from AdaGrad to ADAM and the initial learning rate was set to 0.0001. This architecture adheres to the guidelines set forth in the __Approach__ section and resulted in a significant improvement in the performance of the model. After training the model for 50,000 steps over the course of approximately 23 hours, the model achieved 77.5% accuracy on the evaluation data set. 

Further review of references [14, 15] indicates that the performance of this model might be further enhanced by the inclusion of an additional set of conv/conv/pool/dropout layers to convolve the 8x8 output of the __dropout2__ layer. The new convolutinal layers would make use of either 96 or 128 3x3 filters in an attempt to extract even more features from training data. Unfortunately, time constraints did not allow for the evaluation of such an approach.

### CIFAR-100

Like the CIFAR-10 data set, CIFAR-100 is comprised of 60,000 relatively complex color images. However, whereas CIFAR-10 images are limited to 10 possible classification, CIFAR-100 has 100 possible classifications for each image. The possible classifications are shown below

![Image](CIFAR100-Classes.png)

CIFAR-100 must be downloaded from the website indicated in [11]. Once downloaded, the user must decompress the resulting file and make its two component files (one containing 50,000 images for training; the other containing 10,000 images for testing) accessible to python. The user must then apply the pickle.load() / reshape procedure as outlined above for the CIFAR-10 data set to transform the downloaded data into a format that will work within the Tensorflow environment.

The architecture of the CNN model for the CIFAR-100 data set is shown below.

![Image](CIFAR100-Model.png)

The architecture is very similar to that used for CIFAR-10. However, given the increase in complexity of CIFAR-100's 100 possible classifications, the number of elements for the fully connected dense layers has been increased in an attempt to extract additional features from the training data. A first attempt at training a CNN with this architecture made use of an ADAM optimizer set with an initial learning rate of 0.001. However, no substantive convergence of the loss function was observed using those parameters. The initial learning rate was then reduced to 0.0001 and this enabled the eventual convergence of the loss function.

The source code for this model can be found at the following Github link:

https://github.com/jtopor/CUNY-MSDA-661/blob/master/CIFAR-100/TF-Layers-CIFAR-100-v2.py

After training the model for 65,000 steps over the course of approximately 25 hours, the model achieved 100% accuracy on the training set with a loss 0.392 and 47% accuracy with a loss of 2.464 on the evaluation data set. These results indicate that the model may be overfitting the training data to some degree. As with the CIFAR-10 data set, the performance of this model might be further enhanced by the inclusion of an additional set of conv/conv/pool/dropout layers to convolve the 8x8 output of the __dropout2__ layer. The new convolutinal layers would make use of either 96 or 128 3x3 filters in an attempt to extract even more features from training data. Unfortunately, time constraints did not allow for the evaluation of such an approach.


### Labeled Faces in the Wild

The Labeled Faces in the Wild (LFW) data set has been widely used for purposes of training CNN's for facial recognition systems due to its public availability and concise classification labeling. Unfortunately, CNN models derived from the LFW data set have frequently suffered from overfitting. According to a recent article in IEEE Spectrum [16], such overfitting appears to have been a widespread problem for virtually all widely used facial recognition algorithms that were developed using the LFW dataset, including Google's FaceNet. The overfitting may be a result of the relatively limited 13,575 item breadth of the data set. However, that does not negate LFW's value as a learning and CNN model evaluation tool.

The LFW data set can be accessed directly within python via the __sklearn__ package. The python code snippet below shows how a pre-built function within __sklearn__ can be used to load the LFW data set and simultaneously resize the original 250x250 images to a size of 64x64. (*NOTE: this snippet will only load images pertaining to any individual represented by at least 30 images within the data set. If the full data set is desired, simply exclude the __min_faces_per_person__ argument.*)


```python
# load the LFW data loader from the sklearn.datasets package
from sklearn.datasets import fetch_lfw_people

# load images for individuals w/ 30+ images and produce centered 64x64 images from orig. 250x250 images
lfw_people = fetch_lfw_people(min_faces_per_person=30, 
                              slice_ = (slice(61,189),slice(61,189)),
                              resize=0.5, color = True)

# access the images
X = lfw_people.images

# access the class labels
y = lfw_people.target
```

The original 250x250 images are resized for performance purposes: the larger the images within a data set are, the more computationaly intensive the convolutional layers of a CNN will be due to the fact that the filters within each convolutional layer will need to convolve over each image pixel. By reducing the images from 250x250 to 64x64 we significantly reduce the computational requirements of the CNN. The data set is also subsetted to extract images of any individual that is 
represented in at least 30 images within the LFW data set. This subsetting yields a total of 2370 images with 34 possible classifications, which are then divided up into training and evaluation subsets, with the training set having 1777 images and 593 images set aside for evaluation.

The architecture used for the CNN for the 64x64 LFW images is shown below.

![Image](LFW-64.png)

The source code for this model can be found at the following Github link:

https://github.com/jtopor/CUNY-MSDA-661/blob/master/LFW-CNN/TF-Layers-LFW-Github.py

The diagram shows that the architecture makes use of three sets of convolution/convolution/pooling layers followed by a single fully connected layer comprised of 1,024 elements. Applying an AdaGrad optimizer with an initial learning rate of 0.001 led to convergence of the loss function within 10,000 steps over the course of approximately 6 hours. Applying trained CNN to the evaluation subset showed the model to be 79.1% accurate with a loss of 1.09.

While these results were encouraging, they were obtained using a relatively small subset of the LFW data set. In an attempt to expand the usage of the data set while maintaining a relatively reasonable amount of time required for model training, a second subsetting of LFW was extracted for all individuals within the data set having at least 14 images and all extracted images being resized to 32x32. This yielded a total of 3,735 images across 106 possible classifications, with 2801 images used for model training and 934 images used for model evaluation. The structure of the model was adjusted to account for the reduced image size as well as the need to extract additional features in light of the more then threefold increase in the number of possible classifications. The revised model is shown below.

![Image](LFW-32.png)

The source code for this revised model can be found at the following Github link:

https://github.com/jtopor/CUNY-MSDA-661/blob/master/LFW-CNN/TF-Layers-LFW-32x32.py

As we can see in the diagram, there are 2 sets of convolution/convolution/polling/dropout layers followed by two fully connected dense layers, one with 1,024 elements and one with 512 elements. Each dense layer is followed by a dropout layer with a 60% retention ratio. Applying an AdaGrad optimizer during model training failed to produce convergence in the loss function. As such, an ADAM optimizer was invoked with an initial learning rate of 0.0001. This lead to convergence of the loss function within 10,200 steps in only two hours. The resulting CNN proved to be 71.3% accurate when applied to the evaluation data with a loss of 1.575.

As with the CIFAR data sets, the performance of this model might be further enhanced by the inclusion of an additional set of conv/conv/pool/dropout layers to convolve the 8x8 output of the __dropout2__ layer. The new convolutinal layers would make use of either 96 or 128 3x3 filters in an attempt to extract even more features from training data. Unfortunately, time constraints did not allow for the evaluation of such an approach.

### CASIA WebFace

Unlike the MNIST, CIFAR, and LFW data sets, the CASIA WebFace (CWF) data set requires the prospective user to obtain formal permission for its download and usage. Once such permission is obtained, the user will be allowed to download a set of very large compressed files which must subsequently unpacked and made available for access by python. With nearly half a million 250x250 images, CWF can prove to be unwieldy within a resource-constricted hardware/software environment (e.g., most Windows-based laptops and desktop PC's). To address this issue, a python script was developed that allows a user to specify a desired number of color images to be subsetted from CWF for use with Tensorflow. The source code for this script can be found at the following Github link:

https://github.com/jtopor/CUNY-MSDA-661/blob/master/CASIA-WebFace/casia-TF-dataprepper.py

The script does the following:

1. Extract the desired number of color images and their associated classification labels from a user's local CWF directory structure;

2. If desired, resize each image to a user-specified dimensionality (e.g., 64x64, 32x32, etc.);

3. Reformat the color images for use within Tensorflow;

4. If desired, automatically split the extracted images into training and evaluation subsets according to a user specified split parameter;

5. Create a python __dict__ object containing the images, classification labels, and number of possible classifications for each extracted subset (e.g.,training, evaluation);

6. Write the resulting __dict__ object(s) to a local directory using python's __pickle.dump()__ function.

A user can then easily load the resulting pre-built, Tensorflow-compatible __dict__ objects into their Tensorflow application. An example of how to load and unpack one of these __dict__ objects is shown in the following code snippet:


```python
# how to load dict object from disk in your Tensorflow program
with open('your_path/casia_training64x64.pickle', 'rb') as handle:
    b = pickle.load(handle)

# to access images + labels + number of possible classifications in dict object
train_data = b["Images"]
train_labels = b["Labels"]
n_classes = b["NumClasses"]
```

Details on the exact mechanics of the script can be found within the Github link provided above. The script is heavily documented and should be fairly easy to understand.

#### A CNN Using 64x64 CWF Images

The script was first used to create a subset of 50,000 64x64 CWF images with 524 possible classifications, with 40,000 images set aside for model training and 10,000 images set aside for model evaluation. The structure of the model created to make use of this subset is shown below.

![Image](CWF-64.png)

As we can see in the diagram above, the structure of this CNN model is somewhat different from those that were used on the previously discussed data sets due to the much large number of possible classifications (i.e., 524 vs. 106 or less for all of the other data sets). In particular, we cam see the number of filters used within the convolutional layers increasing from 32 to 128 as we move away from the input layer and toward the output layer. Furthermore, we see a fourth pairing of convolutional layers inserted between the third pooling layer and the only fully connected dense layer. These additional convolutional layers were added in an attempt to extract additional features from the data set. The source code for this model can be found at the following Github link:

https://github.com/jtopor/CUNY-MSDA-661/blob/master/CASIA-WebFace/TF-Layers-Casia-64x64.py

Initially, dropout layers were included after each of the three pooling layers, as was a second full connected dense layer. However, their inclusion precluded the loss function from converging during model training after 15,000 steps executed over the course of more than 30 hours. Additionally, use of an ADAM optimizer failed to produce convergence in the loss function, which led to the implementation of an AdaGrad optimizer. Initial learning rates of 0.001 and 0.0001 also failed to produce convergence in the loss function. After removing the aforementioned dropout layers, removal of the second fully connected layer, switching the optimizer from ADAM to AdaGrad, and specifying an initial learning rate of 0.01, the loss function proceeded to converge within 7,000 steps executed over the course of more than 20 hours. The resulting model proved to be 100% accurate on the training data with a loss of 0.063. However, when applied to the evaluation data the model proved to be only 40.29% accurate with a loss of 5.19. These results are likely an indication of the model overfitting relative to the training data. Unfortunately, due to both time and computing resource limitations, adjustments to the model that might have alleviated the apparent overfitting (e.g., strategically reintroducing dropout layers or a differently sized second dense layer, etc.) could not be implemented.


#### A CNN Using 32x32 CWF Images

In light of the apparent overfitting seen with the 64x64 CWF subset, an attempt was made to expand the usage of the data set while maintaining a relatively reasonable amount of time required for model training by extracting a new subset of 100,000 32x32 CWF images. The resulting subset was split 75%/25% between model training and model evaluation sets, and covered a total of 1,230 possible classifications (a more than twofold increase over the number of classifications for the 64x64 subset). The CNN model was adjusted to account for the reduced 32x32 size of the input data and the number of filters used within the convolutinal layers was increased in an attempt to extract additional features from the input data. Furthermore, a dropout layer was introduced after the second pooling layer and the retention ratio of the dropout layer following the fully connected layer was reduced to 50%. The revised model is shown below.

![Image](CWF-32.png)

The source code for this revised model can be found at the following Github link:

https://github.com/jtopor/CUNY-MSDA-661/blob/master/CASIA-WebFace/TF-Layers-Casia-32x32-v2.py

During training, the loss function proceeded to converge within 17,000 steps executed over the course of more than 17 hours. The resulting model proved to be 100% accurate on the training data with a loss of 0.393. However, when applied to the evaluation data the model proved to be only 40% accurate with a loss of 4.42. These results are likely an indication of the model overfitting relative to the training data. In an attempt to address the apparent overtraining, dropout layers were added to the model following the 'ppol1' and 'conv6' layers, each with a retention ratio of 75%. Repeating the 17,000 step training process with that revised version of the model yielded an improvement in the loss metric to 3.874 when applied to the evaluation subset but offered no improvement in classification accuracy.

## Transfer Learning

One of the stated objectives of this study was to attempt to use Tensorflow for purposes of implementing "transfer learning",  wherein components of a CNN trained for one task are re-purposed for another task without the need for re-training of the components extracted from the original CNN. For example, we could theoretically take the results of our CIFAR-10 model training efforts and simply replace the output layer with a new output layer that is specific to, say, the LFW data set. Each of the hidden CNN layers between the input layer and the output layer would remain unchanged, with their CIFAR-10 modeling results remaining "frozen" while the new model is retrained to allow it to adapt to the new input and output layers only. Such an approach can drastically reduce the amount of time required for training a model for a new task since much of the required computational effort required for training the hidden layers becomes unnecessary. 

For purposes of this study, our transfer learning efforts were focused on attempts to reuse the trained hidden layers of a CASIA WebFace facial recognition CNN model (either the 64x64 or 32x32 version) for purposes of properly recognizing/classifying facial images contained within the Labeled Faces in the Wild data set. From a CNN toolset standpoint, Tensorflow does, in fact, suggest a methodology for the implementation of transfer learning within the Tensorflow environment. However, the methods they suggest are not in conformance with the __TF Layers__ API they are now promoting as a preferred method for the implementation of CNN's. For example, Tensorflow offers the __tf.Saver()__ function as a tool to save and restore models:

https://www.tensorflow.org/api_docs/python/tf/train/Saver

To use it, the user is required to first explicitly activate a Tensorflow session using the __tf.Session()__ function, after which a model may be saved for future usage. Then, when reuse is desired at some future point, the user must then reconstruct the model in Tensorflow, activate another instance of  __tf.Session()__, and theoretically use the __tf.Saver()__ function to then restore the model. Once restored, the user is supposed to be able to replace one or more layers from the model with new layers that are to be trained for a task different from that of the original CNN. However, this approach directly conflicts with the constructs of the __TF Layers__ API in that __TF Layers__ requires no explicit invocation of a Tensorflow session. Furthermore, __TF Layers__ will reload a CNN model from a previously saved checkpoint without use of the __tf.Saver()__ function. These conflicts seem to preclude the use of any anecdotal suggestions as to how transfer learning might be implemented on a user-built Tensorflow CNN that has been constructed using the __TF Layers__ API. Tensorflow provides very scant guidelines for applying transfer learning: while it does provide an outline for how to make use of ImageNet's pre-built Inception CNN for other image classification tasks (see https://www.tensorflow.org/versions/r0.12/how_tos/image_retraining/), no guidance is offered as to how a user might perform such an operation on a user-built CNN. 

Additional guidance was also sought via a recent textbook [17]. However, the transfer learning example provided therein does not conform with the __TF Layers__ API, and multiple attempts at adapting it to a CNN created with __TF Layers__ proved unsuccessful. As a result, the transfer learning objective proved to be unachievable.

## Conclusion

This independent study provided an opportunity for the author to develop a thorough understanding of the concepts that underlie CNN theory while also acquiring "hands-on" experience with the design and development of CNN's using Google's Tensorflow development platform. A review of relevant CNN research and theory indicated that while neural networks in general, and CNN's specifically, remain a highly empirical field of study, a few key concepts can be articulated:

1. For image classification tasks, deeper CNN's that make use of small convolutional filter sizes appear to outperform CNN's that are either less deep or that rely on relatively larger size convolutional filters.

2. Adding dropout layers to a CNN model can help to reduce the likelihood of a CNN model overfitting its training data;

3. No single optimizing algorithm (i.e., SGD, ADAM, AdaGrad, etc.) is "the best" to use for any particular data set. Selection of an effective optimizer generally requires trial and error testing with whatever data you are using.

4. No single initial learning rate will guarantee convergence of a CNN's loss function during model training. If you choose too small of an initial learning rate, your model might converge so slowly as to prove infeasible for practical purposes. Conversely, selecting a value that is relatively large may prevent a CNN model from ever converging in any meaningful manner.

5. CNN model developers must be prepared to devote a fair amount of time toward experimenting with both model structure and model parameters if an effective CNN model is to be achieved. CNN development truly is an empirical domain.

6. The greater the complexity of the classification task, the harder it will be to design and implement an effective CNN model.

7. Overfitting can easily occur when training a CNN for a relatively complex classification task.

The chart shown below provides a brief summary of the models discussed herein. As we can see, each of the five data sets used for model development required varying amounts of time for model training, and no single optimizer proved to be effective across all of the data sets.

![Image](Summary.png)

We can also see that the models that we suspect of suffering from overfitting (i.e., CIFAR-100; the CASIA models) vastly underperformed the models that did not appear to suffer from overfitting. As such, a possible avenue for future research using these models might include investigating ways in which that overfitting might be alleviated, e.g., via data augmentation or changes to the architecture of the models. Additional work might also include investigating ways in which the models developed herein might be trained and maintained within a higher performance virtual environment (e.g., a cloud-based high performance virtual machine) or via the use of high speed graphics processing units (GPU's); a major constraint on this work proved to be the computational limitations of the author's local computing environment, comprised as it was of a single Windows 10 laptop whose GPU underperformed its CPU.


## References

1. Stanford CS231n Materials: http://cs231n.github.io/

2. Tensorflow Website: https://www.tensorflow.org/

3. Tensorflow: MNIST for Machine Learning Beginners: https://www.tensorflow.org/get_started/mnist/beginners

4. A Guide to TF Layers: Building a Convolutional Neural Network: https://www.tensorflow.org/tutorials/layers

5. Deep MNIST for Experts: https://www.tensorflow.org/get_started/mnist/pros#build_a_multilayer_convolutional_network

6. Dong Yi, Zhen Lei, Shengcai Liao and Stan Z. Li. "Learning Face Representation from Scratch"", arXiv:1411.7923v1 [cs.CV], 2014. (https://pdfs.semanticscholar.org/853b/d61bc48a431b9b1c7cab10c603830c488e39.pdf)

7. Garcia, Christophe, and Delakis, Manolis. "Convolutional face finder: A neural architecture for fast and robust face detection", IEEE Transactions on Pattern Analysis and Machine Intelligence 26(11):1408 - 1423 · December 2004

8. Duffner, Steffan. "Face Image Analysis With Convolutional Neural Networks", Doctoral Dissertation, Albert Ludwigs University of Freiburg, Breisgau, Germany, 2007 (https://pdfs.semanticscholar.org/dbb7/f37fb9b41d1aa862aaf2d2e721a470fd2c57.pdf)

9. MNIST: http://yann.lecun.com/exdb/mnist/

10. CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

11. CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html

12. Labeled Faces in the Wild ('LFW'): http://vis-www.cs.umass.edu/lfw/index.html

13. CASIA WebFace Database: http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html

14. Jason Brownlee. "Object Recognition with Convolutional Neural Networks in the Keras Deep Learning Library" : http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/

15. Parneet Kuar. "Convolutional Neural Networks (CNN) for CIFAR-10 Dataset" : http://parneetk.github.io/blog/cnn-cifar10/

16. http://spectrum.ieee.org/computing/software/finding-one-face-in-a-million

17. Aurelian Geron. "Hands-On Machine Learning with Scikit-Learn & Tensorflow", 2017 , O'Reilly Media Inc., pp 286-290.


