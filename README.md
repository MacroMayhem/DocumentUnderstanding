# Document Understanding
An exploratory work on detecting, recognizing and categorizing texts in document images 

## Introduction
Before diving into the implementation it is really important to understand the problem we are trying to solve and define the do's and don'ts of the system. Moving from here we want to mention, maybe in some hierarchy, checkpoints we want to achieve. Also, It is good to develop smaller models for smaller tasks and ensemble them instead of building a monolithic model which sovles everything.

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
  * others (Anything which is doesn't fall into above mentioend categories)

### Data
We describe the data we have for the job and then move ahead on creating a pipeline with the data at hand. The appraoch we use depends on the kind and amount of data we have. For this task we don't have any annotated data. We can either go ahead and generate synthetic data, or use data from a similar domain. For our approach we will be using synthetic data at parts where it's feasible to generate one and use public data when it's easily available.

##### Details

1. Synthetic generation of datasets corresponding to **Dates, Locations, Company Name, IDs, Goods, Others**
2. If possible collection of easily available datasets mentioned above and mixing both

*Future Work* For document images we currently don't have an easily available dataset. Might have to stick to augmenting it
*Future Work* We gather a corpus of words for each cateogry mentioned in the problem statement. 
*Future Work* We will synthesize artificial data in some intelligent way using the corpuses.


### Problem formulation
We can think of the final task of categorization as Named Entity Recognition, where we tag different strings encountered as Person, Organisation, Dates etc. There are can be lots of  approaches we can and should try out. But intial attempt should be exploring existing techniques before coming up with a complicated solution on our own. We will break the approach into different pipelines and chalk out our expectations from the modules and use readily avaiable methods wherever applciable. 
 

### Pipeline
Let's hypothesize a pipeline we think will achieve the goal. I am listing these down in chronological order in which the processing will take place by the system, but, during development we will work on them in a slightly different manner.

1. **Text Detection module**: `Signature: text_detection(Image) -> {(x,y,del_x,, del_y,confidence)}` String is located in the rectangle on the image from pixel(x,y) to pixel(x+del_x,y+del_y). We don't know what the text is but we are very sure(**confidence**) there is some text there. We focus on the granularity at word level i.e. we are interested in knowing regions where there are words and not single characters or sentences. Possible errors the module might create are:
  
  a. **False Positives**: The module might and will detect regions with no text as text
  
  b. **False Negatives**: The module can and most likely will miss detections of texts in the image
  
  c. **Multiple Detection**: Detected a sentence as a string. For example, *Electronic item* can be detected as a single string even though the granularity we expect during the detection is at word level
  
  We will focus in having a **high recall** as it might be possible to remove false positives later on in the pipeline however, if we miss a positive sample early on in the stage we will never be able to retrieve this infromation. 
  

2. **Text Recognition Module**: `Signature: text_recognition(Image(x,y,del_x,del_y))->{[string, confidence]}` Given a patch of del_x,del_y in the image this module returns the word or sequence of words contained in the patch ordered by some confidence.
Again, this module will be not a perfect module and some possible errors it might face are
  
  a. **Miss-Spelled Words**: A word *burger* can be recognized as a *burgler* or a *bugger*. 
  
  b. **No Detections**: The algorithm might not even find a word, especially for cases when there were no text to begin with.Might even return a random string.

 **Note:** For the exploratory work we are usign the Google Vision API which does the steps 1 and 2 or perhaps some other OCR or a collection of them. 

3. *Future Work* **Alignment Module**: `Signature: alignment([(x,y,word)..]) -> [ordering]` This module provides the most likely ordering of the texts. Sample figure provides the insight on the working. During the initial exploratory work we will drop the idea of implementing it due to the lack of GT. We can comeup with strategy to synthesize the data for example starting with some random layout generation and populating it randomly with corpuses we have, but for the time being we will ignroe this module.

4. **Content Categorization**: `Signature: categorize([Block]) -> [confidence_class_0,class_1,..class_5]` Block can be a single string or a sequence of strings. In this part we will feed the **individual** strings from the **block** obtained in **step 2** to the system and obtain a probability of the assignment variable over the **6** classes (if possible else its a 1-hot encoding). We can then compute the probability of the block for each class as *insert image here*. We can then depending on the values classify the text block multiple classes if need be and this would hint us at breaking and analysing the block even further.


### Content Categorization
##### Data collection
We will be collecting data for [Company names](https://datahub.io/core/nasdaq-listings), [Goods](), [Location](https://github.com/datasets/world-cities), synthesize from [Dates](https://docs.oracle.com/cd/E41183_01/DR/Date_Format_Types.html) 

We will start of with content categorization. The input is a *block(sequence of space separated strings)* 
**About the Data** we have with us necessarily *text* data with us. 
**Approaches for individual classes** 

1. First Approach which comes to mind is using Named Entity Recognition or perhaps at a deeper level as Part of Speech Tagging. Here given a sequence of strings for NER we map each word to Name, Organiztion, Date etc. But it has few problems. We don't know how the sentences will come in for the inference and hence the construction of training data is not well defined. We can also have Out of Vocabulary words as oraganization names, good etc, if we dont have a proper domain corpus and as a result we wont be able to provide good labelling at test.
2. We can use rule based approaches for few of the categories such as Date i.e. construct a regex which matches with some predefined structure of date. This will be a knowledge driven methd and hence if we encounter a new date format the method will most likely fail. 
3. Embedding approaches might potentially fail due to the OOV words during test.
4. For ReceiptID(Random String) we can define a rule of acceptance if a unit length string is not classified as Name, Type, Location or Date. Composing of alphanumerics only.
5. Everything not belonging to the *5* classes goes to Others.

It seems now that we have defined heuristics for 2 classes but are still left with 3 classes. Namely, *Name, Location, Type of Good*. Even though we might get some bad performance on validation/tests NER by Standford is a good way to start.

##### NER
Lets start by analysing the performance of **Name, Location, Type of Good and Date(We can choose a regex or NER)**

| Method | Qualitative assessment |
| --- | --- |
| NLTK NER (Case-Sensitive)|     |


