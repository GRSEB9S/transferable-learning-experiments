# Transferable Learning in Convolutional Neural Networks

# Abstract

Recent advances in deep learning have catapulted multi-layer neural networks to the forefront of the machine learning and computer vision communities. In the past few years, deep neural networks have made remarkable strides in multiple domains, including object recognition, speech recognition, and control (reviewed in reviewed in LeCun et al., 2015 and Lake et al., 2016). In particular, convolutional neural networks (“convnets,” LeCun et al., 1989) have now reached human-level performance on object-recognition tasks (He et al., 2015; Russakovsky et al., 2015).

Despite the rapid pace of progress in deep learning, researchers have only recently begun to consider the question of whether neural networks constitute reasonable models of human cognition beyond limited, superficial similarities to biological neural networks. One key component of human perceptual learning is the ability to rapidly generalize acquired knowledge across categorical domains. Neurological and psychophysical studies confirm that humans quickly learn to recognize novel stimuli, in some cases after just a single exposure (Rutishauser et al., 2006; Carey and Bartlett, 1978; Pinker, 1999). In contrast, modern neural networks require thousands of training iterations on vast datasets in order to match human capacity (Russakovsky et al., 2015).

This discrepancy raises an interesting dilemma. Given that randomly-initialized neural networks must start from zero previous visual experience, while humans typically draw on an extensive body of perceptual knowledge, is it fair to directly compare the performance of humans and neural nets on the same recognition tasks?

In order to shed light on this issue, we conducted a systematic study of neural nets' ability to generalize perceptual knowledge from one object classification task to another. Our research was guided by three main experimental questions:

* To what extent does learning transfer from one classification task domain to another?
* At what layer(s) of the network is transferable knowledge stored?
* What are the layerwise temporal dynamics of learning? In other words, does learning occur in a bottom-up or top-down manner?

For each of these questions, we designed and performed a computational experiment. In Section 3.1, we found that the process of learning can be greatly accelerated through pre-training. These results suggest that the network stores some degree of generalizable perceptual knowledge that can be transferred across image domains. In Section 3.2, we “lesioned” different layers of the net in order to determine where in the network transferable knowledge is stored. We found that lesioning higher layers of the network impaired performance, while lesioning low layers generally had minimal impact. Finally, in Section 3.3, we measured the layerwise evolution of the values of the network’s weights and biases in an attempt to quantify the temporal dynamics of learning. We found that learning occurs in a top-down manner, where higher layers undergo more and earlier changes than lower layers. Taken together, our results support a reverse-hierarchical account of deep learning that is consistent with theories of perceptual learning in the brain.

# Next Steps

* Explore getting the nets to perform the tasks described in the RHT study.
* Train on a richer dataset with multiple categories.
* Explore additional topogrophies.
