# Document Understanding
An exploratory work on detecting, recognizing and categorizing texts in document images 

## Introduction
Before diving into the implementation it is really important to understand the problem we are trying to solve and define the do's and don'ts of the system. Moving from here we want to mention, maybe in some hierarchy, checkpoints we want to achieve. Also, It is good to develop smaller models for smaller tasks and ensemble them instead of building a monolithic model which solves everything.

### Problem Statement
Given an Image(I) of a receipt, we want the system to `categorize/summarize the text-content` present in the image.
For now it's safe to assume 
* I is always a scanned image of text document
* I only contains computer generated text i.e. no hand-written content
* Different classes of interest in the content of the receipts are
  * Date of purchase
  * Name (Company Name)
  * Location 
  * Receipt number (Some ID: 1270348, or ABC123DEF)
  * Physical goods (electronic, food stuff, toys etc.)
  * others (Anything which doesn't fall into the above mentioned categories)

### Data
We describe the data we have for the job and then move ahead on creating a pipeline with the data at hand. The appraoch we use depends on the kind and amount of data we have. For this task we don't have any annotated data. We can either go ahead and generate synthetic data, or use data from a similar domain. For our approach we will be using synthetic data at parts where it's feasible to generate one and use public data when it's easily available.

#### Content Categorization
##### Data Collection

 *  Companies [1](https://www.kaggle.com/dattapiy/sec-edgar-companies-list) [2](https://www.kaggle.com/Eruditepanda/fortune-1000-2018)
 *  [Locations](https://datahub.io/core/world-cities)
 *  [Goods](https://www.kaggle.com/PromptCloudHQ/flipkart-products)

##### Data Cleaning
 
 1. __Companies__ : 
   * Randomly sample 20000 company names from the dataset
   * Remove stopwords, keep only alphanumerics and convert all to lower case
   * Split words on the space character and add individual words to training
 
 2. __Location__ : 
   * Randomly sample 10000 cities from the data 
   * All the country names are added for training
   * Convert all strings to lowercase and remove special chars if necessary
 
 3. __Goods__ :
   * Use the 'product_category_tree' to obtain the hierarchy for every item.
   * Process each element of the hierarchy in a similar fashion. Removal of stop words, lowercasing, and removing special characters
 
##### Feature Respresentation

We utilise [FastText](https://fasttext.cc/) based representation for words. It is better to use over _word2vec_ as it can tolerate _Out of Vocabulary_ words due to it's character level n-gram formulation. We train a _Fasttext_ model which yields __50__ dimensional representation for a word. 
 

### Pipeline

1. **Text Detection module**: `Signature: text_detection(Image) -> {(x,y,del_x,, del_y,confidence)}` String is located in the rectangle on the image from pixel(x,y) to pixel(x+del_x,y+del_y). We don't know what the text is but we are very sure(**confidence**) there is some text there. We focus on the granularity at word level i.e. we are interested in knowing regions where there are words and not single characters or sentences. Possible errors the module might create are:
  
    a. **False Positives**: The module might and will detect regions with no text as text
  
    b. **False Negatives**: The module can and most likely will miss detections of texts in the image
  
    c. **Multiple Detection**: Detected a sentence as a string. For example, *Electronic item* can be detected as a single string even though the granularity we expect during the detection is at word level
  
  We will focus in having a **high recall** as it might be possible to remove false positives later on in the pipeline however, if we miss a positive sample early on in the stage we will never be able to retrieve this infromation. 
  

2. **Text Recognition Module**: `Signature: text_recognition(Image(x,y,del_x,del_y))->{[string, confidence]}` Given a patch of del_x,del_y in the image this module returns the word or sequence of words contained in the patch ordered by some confidence.
Again, this module will be not a perfect module and some possible errors it might face are
  
    a. **Miss-Spelled Words**: A word *burger* can be recognized as a *burgler* or a *bugger*. 
  
    b. **No Detections**: The algorithm might not even find a word, especially for cases when there were no text to begin with.Might even return a random string.

 **Note:** For the exploratory work we are using the Google Vision API which does the steps 1 and 2 or perhaps some other OCR or a collection of them. 

3. *Future Work* **Alignment Module**: `Signature: alignment([(x,y,word)..]) -> [ordering]` This module provides the most likely ordering of the texts. Sample figure provides the insight on the working. During the initial exploratory work we will drop the idea of implementing it due to the lack of GT. We can comeup with strategy to synthesize the data for example starting with some random layout generation and populating it randomly with corpuses we have, but for the time being we will ignroe this module.

4. **Content Categorization**: `Signature: categorize(String) -> [confidence_class_0,class_1,..class_5]`
        
        * Company, Location, GoodsType: We train classifiers for each individually while keeping data from the other two classes as negatives. 
        * Dates: We make use of `parse from dateutil.parser` to check if a string is a date or not
        * Random String: Regex matching using `^[a-zA-Z0-9-/]+$` as the pattern
        * Others: Anything which is not accepted by the previous methods will be classified as `Others`





