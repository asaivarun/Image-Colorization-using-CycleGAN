# Image-Colorization-using-CycleGAN
Image-to-image translation is a class of computer vision problem where the goal is to learn the mapping between an input and output image domain using training sets of images from both domains.

I tried out three different approaches for image colorization task: The first approach involves the use of a cycleGAN architecture to address image colorization as a image-to-image translation problem.  

Second, we explore the use of capsule networks[13] as discriminators for the cycleGAN architecture to perform image colorization. 

I also implement a stochastic version of our GAN architecture to generate multiple color images for a single image. 

To benchmark the performance of our unpaired image-to-image translation framework, we also train a conditionalGAN architecture which performs image colorization in a supervised manner.

GAN : Generative Adversarial Networks composed of two smaller networks called the generator and discriminator. The generator's task is to produce results that are indistinguishable from real data whereas discriminator tries to classify whether a sample came from the generators model distribution or the original data distribution. Both of these sub networks are trained simultaneously until the generator is able to consistently produce results that the discriminator cannot classify.

For detailed analysis refer to https://github.com/asaivarun/Image-Colorization-using-CycleGAN/blob/master/project_zahilsha_ysaraf_saivarun.pdf
