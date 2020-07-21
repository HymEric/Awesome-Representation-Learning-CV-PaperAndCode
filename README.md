# Awesome-Representation-Learning-CV-PaperAndCode
Learning useful representations with the unsupervised or weakly supervised methods is a key challenge in artificial intelligence. The **representation learning (RL)** is helpful for most specific tasks like classification, recognition, detection, image editing, image retrieval, et al. in computer vision area. RL is mainly appeared in the learning good representations for down-stream task, disentangled representation/attributes, VAE, GAN, Flow-based model, image-translation, deep clustering papers. This repo mainly focuses on the lasted development in the RL area.

I hope this repo helps both you and me. If you find some mistakes, other novel or interesting works related to **representation learning**, please don't hesitate to **issue** or **pull request**.

## Contents
<!-- TOC -->

- [1. Related Metrics](#Related-Metrics)
- [2. Related Survey](#Related-Survey) 
- [3. General Representation Learning](#General-Representation-Learning)
- [4. Disentangled Representation](#Disentangled-Representation)
- [5. Disassembling Object Representation](#Disassembling-Object-Representation)
- [6. VAE-based Method](#VAE-based-Method)
- [7. GAN-based Method](#GAN-based-Method)
- [8. Flow-based Method](#Flow-based-Method)
- [9. Image Translation](#Image-Translation)
- [10. Deep Clustering](#Deep-Clustering)
- [11. Resources](#Resources)

<!-- TOC -->

## Related Metrics
- [Evaluating the Disentanglement of Deep Generative Models through Manifold Topology](https://arxiv.org/abs/2006.03680), arXiv2020
- [Theory and Evaluation Metrics for Learning Disentangled Representations](https://arxiv.org/abs/1908.09961), arXiv2019
- [A framework for the quantitative evaluation of disentangled representations](https://openreview.net/forum?id=By-7dz-AZ&noteId=By-7dz-AZ), ICLR2018

## Related Survey
- [Representation learning: A review and new perspectives](https://arxiv.org/abs/1206.5538), PAMI2013, Yoshua Bengio
- [Recent Advances in Autoencoder-Based Representation Learning](https://arxiv.org/abs/1812.05069), arXiv2018


## General Representation Learning
##### In 2020
- [Parametric Instance Classification for Unsupervised Visual Feature Learning](https://arxiv.org/abs/2006.14618), arXiv2020, PIC


- [A Universal Representation Transformer Layer for Few-Shot Image Classification](https://arxiv.org/abs/2006.11702), arXiv2020  
[codes-tensorflow](https://github.com/liulu112601/URT)

- [Self-supervised Learning Generative or Contrastive](https://arxiv.org/abs/2006.08218), arXiv2020

- [Gradients as Features for Deep Representation Learning](https://openreview.net/forum?id=BkeoaeHKDS), ICLR2020  
[codes-pytorch](https://github.com/fmu2/gradfeat20)

- [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029), arXiv2020, SimCLR-v2  
[codes-tensorflow](https://github.com/google-research/simclr)

- [A simple framework for contrastive learning of visual representations](https://arxiv.org/abs/2002.05709), arXiv2020, SimCLR-v1  
[codes-pytorch](https://github.com/sthalles/SimCLR)

- [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297), arXiv2020, MoCo-v2  
[codes-pytorch-unofficial](https://github.com/AidenDurrant/MoCo-Pytorch)

- [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722), CVPR2020, MoCo-v1  
[codes-pytorch](https://github.com/facebookresearch/moco)

- [Self-supervised learning of pretext-invariant representations](http://openaccess.thecvf.com/content_CVPR_2020/html/Misra_Self-Supervised_Learning_of_Pretext-Invariant_Representations_CVPR_2020_paper.html), CVPR2020, PIRL  
[codes-pytorch-unofficial](https://github.com/akwasigroch/Pretext-Invariant-Representations)

- [Prototypical Contrastive Learning of Unsupervised Representations](https://arxiv.org/abs/2005.04966), arXiv2020

- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362), arXiv2020  
[codes-pytorch](https://github.com/HobbitLong/SupContrast)

- [Contrastive representation distillation](https://arxiv.org/abs/1910.10699). ICLR2020  
[codes-pytorch](https://github.com/HobbitLong/RepDistiller)

##### In 2019
- [Learning deep representations by mutual information estimation and maximization](https://arxiv.org/abs/1808.06670), ICLR2019  
[codes-pytorch](https://github.com/rdevon/DIM)

- [Revisiting self-supervised visual representation learning](http://openaccess.thecvf.com/content_CVPR_2019/html/Kolesnikov_Revisiting_Self-Supervised_Visual_Representation_Learning_CVPR_2019_paper.html), CVPR2019

##### In 2018
- [Representation learning with contrastive predictive coding](https://arxiv.org/abs/1807.03748), arXiv2018  
[codes-keras](https://github.com/davidtellez/contrastive-predictive-coding)

- [Unsupervised feature learning via non-parametric instance discrimination](http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.html), CVPR2018  
[codes-pytorch](https://github.com/zhirongw/lemniscate.pytorch)


- [Contrastive learning of emoji-based representations for resource-poor languages](https://arxiv.org/abs/1804.01855)

##### In 2016
- [Order-Embeddings of Images and Language](https://arxiv.org/abs/1511.06361),  ICLR2016  
[codes-theano](https://github.com/ivendrov/order-embedding)

##### In 2015
- [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192), ICCV2015  
[code-caffe](https://github.com/cdoersch/deepcontext)

## Disentangled Representation
##### In 2020
- [ICAM: Interpretable Classification via Disentangled Representations and Feature Attribution Mapping](https://arxiv.org/abs/2006.08287), arXiv2020  
[codes](https://github.com/CherBass/ICAM)

##### In 2019
- [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359), ICML2019
- [Emerging disentanglement in auto-encoder based unsupervised image content transfer](https://arxiv.org/abs/2001.05017), ICLR2019  
[codes-pytorch](https://github.com/oripress/ContentDisentanglement)

- [Are Disentangled Representations Helpful for Abstract Visual Reasoning?](http://papers.nips.cc/paper/9570-are-disentangled-representations-helpful-for-abstract-visual-reasoning), NeurIPS2019

- [Hyperprior induced unsupervised disentanglement of latent representations](https://www.aaai.org/ojs/index.php/AAAI/article/view/4185), AAAI2019  
[codes-tensorflow](https://github.com/clear-nus/CHyVAE)


##### In 2018
- [Learning disentangled joint continuous and discrete representations](http://papers.nips.cc/paper/7351-learning-disentangled-joint-continuous-and-discrete-representations), NeurIPS2018

- [Unsupervised representation learning by predicting image rotations](https://arxiv.org/abs/1803.07728), ICLR2018  
[codes-pytorch](https://github.com/gidariss/FeatureLearningRotNet)

- [A Two-Step Disentanglement Method](http://openaccess.thecvf.com/content_cvpr_2018/html/Hadad_A_Two-Step_Disentanglement_CVPR_2018_paper.html), CVPR2018  
[codes-keras](https://github.com/naamahadad/A-Two-Step-Disentanglement-Method)

- [Disentangling by factorising](https://arxiv.org/abs/1802.05983), ICML2018  
[codes-pytorch](https://github.com/1Konny/FactorVAE)

- [Isolating sources of disentanglement in variational autoencoders](http://papers.nips.cc/paper/7527-isolating-sources-of-disentanglement-in-variational-autoencoders), NeurIPS2018  
[codes-pytorch](https://github.com/rtqichen/beta-tcvae)

- [Life-long disentangled representation learning with cross-domain latent homologies](http://papers.nips.cc/paper/8193-life-long-disentangled-representation-learning-with-cross-domain-latent-homologies), NeurIPS2018

- [A Spectral Regularizer for Unsupervised Disentanglement](https://arxiv.org/abs/1812.01161), arXiv2018

- [Visual object networks: Image generation with disentangled 3D representations](http://papers.nips.cc/paper/7297-visual-object-networks-image-generation-with-disentangled-3d-representations), NeurIPS2018

- [Understanding disentangling in beta-VAE](https://arxiv.org/abs/1804.03599), arXiv2018

- [Disentangling the independently controllable factors of variation by interacting with the world](https://arxiv.org/abs/1802.09484), arXiv2018

- [Dual swap disentangling](http://papers.nips.cc/paper/7830-dual-swap-disentangling), NeurIPS2018

##### In 2017
- [Unsupervised learning of disentangled representations from video](http://papers.nips.cc/paper/7028-unsupervised-learning-of-disentangled-representations-from-video), NeurIPS2017  
[codes-pytorch](https://github.com/ap229997/DRNET)


## Disassembling Object Representation
##### In 2020
- [Disassembling Object Representations without Labels](https://arxiv.org/abs/2004.01426), arXiv2020

- [Learning to Manipulate Individual Objects in an Image](https://arxiv.org/abs/2004.05495), CVPR2020

##### In 2019
- [Monet: Unsupervised scene decomposition and representation](https://arxiv.org/abs/1901.11390), arXiv2019  
[codes-pytorch](https://github.com/baudm/MONet-pytorch)

- [Multi-object representation learning with iterative variational inference](https://arxiv.org/abs/1903.00450), ICML2019  
[codes-pytorch](https://github.com/zhixuan-lin/IODINE)

- [Object Discovery with a Copy-Pasting GAN](https://arxiv.org/abs/1905.11369), arXiv2019  
[codes-tensorflow](https://github.com/wtupc96/Copy-Pasting-GAN)

- [GENESIS: Generative Scene Inference and Sampling with Object-Centric Latent Representations](https://arxiv.org/abs/1907.13052), ICLR2019  
[codes-pytorch](https://github.com/applied-ai-lab/genesis)

- [Stacked capsule autoencoders](http://papers.nips.cc/paper/9684-stacked-capsule-autoencoders), NeurIPS2019  
[codes-pytorch](https://github.com/phanideepgampa/stacked-capsule-networks)

- [LAVAE: Disentangling Location and Appearance](https://arxiv.org/abs/1909.11813), arXiv2019

- [Unsupervised object segmentation by redrawing](http://papers.nips.cc/paper/9434-unsupervised-object-segmentation-by-redrawing), NeurIPS2019  
[codes-pytorch](https://github.com/mickaelChen/ReDO)

##### In 2018
- [Relational neural expectation maximization: Unsupervised discovery of objects and their interactions](https://arxiv.org/abs/1802.10353), ICLR2018  
[codes-tensorflow](https://github.com/sjoerdvansteenkiste/Relational-NEM)

##### In 2016
- [Attend, infer, repeat: Fast scene understanding with generative models](http://papers.nips.cc/paper/6230-attend-infer-repeat-fast-scene-understanding-with-generative-models), NeurIPS2016  
[codes-pytorch](https://github.com/Abishekpras/Attend-Infer-Repeat---Pytorch)

## VAE-based Method
##### In 2019
- [Spatial broadcast decoder: A simple architecture for learning disentangled representations in vaes](https://arxiv.org/abs/1901.07017), arXiv2019

##### In 2018
- [Disentangling Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1812.02833), ICML2019

- [Disentangled Sequential Autoencoder](https://arxiv.org/abs/1803.02991), ICML2018  
[codes-pytorch](https://github.com/yatindandi/Disentangled-Sequential-Autoencoder)

##### In 2017
- [beta-VAE Learning Basic Visual Concepts with a Constrained Variational Framework](https://pdfs.semanticscholar.org/a902/26c41b79f8b06007609f39f82757073641e2.pdf), ICLR2017  
[codes-pytorch](https://github.com/1Konny/Beta-VAE)

##### In 2014
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), ICLR2014  
[codes-keras](https://github.com/bojone/vae), [codes-pytorch](https://github.com/nitarshan/variational-autoencoder)


## GAN-based Method
##### In 2020
- [Transformation GAN for Unsupervised Image Synthesis and Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Transformation_GAN_for_Unsupervised_Image_Synthesis_and_Representation_Learning_CVPR_2020_paper.html), CVPR2020

##### In 2019
- [Large scale adversarial representation learning](http://papers.nips.cc/paper/9240-large-scale-adversarial-representation-learning), NeurIPS2019  
[codes-tensorflow](https://github.com/LEGO999/BigBiGAN-TensorFlow2.0)

##### In 2017
- [PixelGAN autoencoders](http://papers.nips.cc/paper/6793-pixelgan-autoencoders), NeurIPS2017 

- [Adversarial feature learning](https://arxiv.org/abs/1605.09782), ICLR2017  
[codes-theano](https://github.com/jeffdonahue/bigan)

##### In 2016
- [Infogan: Interpretable representation learning by information maximizing generative adversarial nets](http://papers.nips.cc/paper/6399-infogan-interpretable-representation), NeurIPS2016  
[codes-tensorflow](https://github.com/openai/InfoGAN)

##### In 2015
- [Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/abs/1511.06434), arXiv2015  
[codes-tensorflow](https://github.com/jazzsaxmafia/dcgan_tensorflow)


## Flow-based Method
##### In 2019
- [Flow++: Improving flow-based generative models with variational dequantization and architecture design](https://arxiv.org/abs/1902.00275), ICML2019

##### In 2018
- [Glow Generative flow with invertible 1x1 convolutions](http://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-con), NeurIPS2018  
[codes-tensorflow](https://github.com/openai/glow)

##### In 2017
- [Masked autoregressive flow for density estimation](http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation), NeurIPS2017  
[codes-tensorflow-unofficial](https://github.com/johnpjust/MAF_GQ_images_tf20)

##### In 2016
- [Improved variational inference with inverse autoregressive flow](http://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow), NeurIPS2016  
[codes-theano](https://github.com/openai/iaf)

## Image Translation
##### In 2020
- [High Resolution Face Age Editing](https://arxiv.org/abs/2005.04410), arXiv2020  
[codes-pytorch](https://github.com/InterDigitalInc/HRFAE)

- [Rethinking the Truly Unsupervised Image-to-Image Translation](https://arxiv.org/abs/2006.06500), arXiv2020  
[codes-pytorch](https://github.com/clovaai/tunit)

##### In 2018
- [Diverse Image-to-Image Translation via Disentangled Representations](http://openaccess.thecvf.com/content_ECCV_2018/html/Hsin-Ying_Lee_Diverse_Image-to-Image_Translation_ECCV_2018_paper.html), ECCV2018  
[codes-tensorflow](https://github.com/taki0112/DRIT-Tensorflow)

- [Image-to-image translation for cross-domain disentanglement](http://papers.nips.cc/paper/7404-image-to-image-translation-for-cross-domain-disentanglement), NeurIPS2018  
[codes-tensorflow](https://github.com/agonzgarc/cross-domain-disen)


## Deep Clustering
##### In 2019
- [Invariant information clustering for unsupervised image classification and segmentation](http://openaccess.thecvf.com/content_ICCV_2019/html/Ji_Invariant_Information_Clustering_for_Unsupervised_Image_Classification_and_Segmentation_ICCV_2019_paper.html), ICCV2019  
[codes-pytorch](https://github.com/xu-ji/IIC)
- [Deep Spectral Clustering using Dual Autoencoder Network](https://arxiv.org/abs/1904.13113), CVPR2019  
[codes-tensorflow](https://github.com/xdxuyang/Deep-Spectral-Clustering-using-Dual-Autoencoder-Network)

##### In 2018
- [Deep Clustering for Unsupervised Learning of Visual Features](http://openaccess.thecvf.com/content_ECCV_2018/html/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.html), ECCV2018  
[codes-pytorch](https://github.com/facebookresearch/deepcluster)

- [Clustering with Deep Learning: Taxonomy and New Methods](https://arxiv.org/abs/1801.07648), arXiv2018  
[codes-theano](https://github.com/ElieAljalbout/Clustering-with-Deep-learning)


## Resources
- [disentangled-representation-papers](https://github.com/sootlasten/disentangled-representation-papers), update 13 months ago
- [Self-supervised learning and computer vision](https://www.fast.ai/2020/01/13/self_supervised/), 13 Jan 2020 
- [Self-Supervised Representation Learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#momentum-contrast), 10 Nov 2019


