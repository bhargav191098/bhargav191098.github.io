---
title: "Law of large numbers"
description: "Yes, the very very basic but beautiful law"
pubDate: "Sep 25 2023"
heroImage: "/cover_post4.jpg"
---

Today, I want to talk about this very fundamental but <i>beautiful</i> law.

## Law of Large numbers: 
It essentially deals with the results of sampling large amounts of data repeatedly. The law states that if we repeatedly sample from a population the mean of the variable in the sampled set converges to the expected value of the variable in the population as the number of samples increases. The expected value of a random variable the average value it could attain. The law of large numbers ties up this value pertaining to a population to the mean of the sampled set. Through this blog post, I want to programmatically check this law in action.

## What is sampling?

Consider you buy one of those Trader Joe's multiflavoured cheesecake. Each time you try one section of the cheesecake, you can form a different opinion. You base all your opinions on the slice you picked but is that slice a true representative of the entire cheesecake? This process of taking a sample of the cheesecake is essentially sampling ;) You base your opinions on the sample you just took. How much this slice varies from the entire cheesecake constitutes the sampling error. We'll take about sampling error and some curious tests in another blog post.

## Sampling in action : 

We'll pull out random samples from the normal distribution.

        mu,sigma = 0,0.1
        sample_means = []
        expected_value = []
        sample_sizes = [i for i in range(10,10000)]
        for sample_size in sample_sizes:
            s_i = np.random.normal(mu,sigma,sample_size)
            sample_mean_i = np.mean(s_i)
            sample_means.append(sample_mean_i)
            expected_value.append(mu)

In the above code: we sample from a normal distribution with <b>mean 0</b> and <b>standard deviation 0.1.</b> The expected value of a random variable following Normal distribution is technically its mean which we have set as 0. So if we keep taking samples from this normal distribution and take its mean, as the number of samples increases, it should converge to 0.

![normal distribution](/normal_distribution.png)

Voila!! The sample converges around the expected value. <br>

Okay this is sampling from a distribution we know before hand. I want to play around with a real life problem and plot the distribution. Let's roll a die!

## Roll 'em:

We'll program rolling a die. Random function to the rescue! Before we get to that, I need to recall some high school math. The expected value of a random value is defined as follows : 
        $ E(X) = \sum x.P(x)$ where x is the possible values of the random variable X.

We are considering an unbiased die. The probability of any of the 6 numbers being the top face is the same and it is 1/6.

    E(X) = 1/6 + 2/6 + 3/6 + 4/6 + 5/6 + 6/6 
    E(X) = 3.5

Cool, the E(X) is sorted. Let's sample from a die and steadily increase the sample space and watch the mean of the sample converge onto this Expected Value.

        '''
        This function simulates rolling a die.
        '''
        def roll_die():
            face_up = random.randint(1,6)
            return face_up
        '''
        This function simulates sampling.
        '''
        def sample_part_of_population(sample_size):
            sample_space = []
            for i in range(sample_size):
                sample_space.append(roll_die())
            return sample_space

        '''
        We'll create sample set sizes using this function.
        '''
        for size in range(1,5000,15):
            sample_sizes.append(size)


        for index in range(len(sample_sizes)):
            sample_size = sample_sizes[index]
            sample = sample_part_of_population(sample_size)
            sample_sum = sum(sample)
            average_value = sample_sum/sample_size
            sample_means.append(average_value)
            sample_expectations.append(expectation())

Let's plot these results.

![Mean of samples](/law_of_large_numbers_excpected_value.png)

## Extension: 

This law can be extended to another cool result. Okay so theoretically what is the probability of getting a 4 or 6 on the top face of the die? 1/6, right? No matter how many times we do it, right? Let's keep that knowledge aside for sometime and define the probability empirically ie. as we see the favourable outcomes, we update the probability based on history.Back to sampling and checking for convergence, baby!

![p hat of getting 4 and 6](/getting_4_6_with_labels.png)

Ooh! The probability of getting a 4 and probability of getting a 6 is converging onto 1/6. It smells like Law of large numbers. Let me draw some bridges for this. First off, the underlying distribution of the die experiment is called Bernoulli Distribution. This distribution is a special case of the normal distribution with n=1. A single trail experiment! We took a roundabout way of getting there and your high school math teacher was smart!  By sampling, we saw $\hat{p}$, the empirical probability converge onto the probability we know as 1/6! 

This law is really important to inferential statistics! Everytime we make estimates from a sample and generalize that solution for a population we are betting on this law to work out! :)



