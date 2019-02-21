# DocumentUnderstanding
An exploratory work on detecting, recognizing and understanding texts in document images 

## Introduction
Before diving into the implementation it is really important to understand the problem we are trying to solve and define the do's and don'ts of the system. Moving from here we want to mention, maybe in some hierarchy, checkpoints we want to achieve. Also, It is good to develop smaller models for smaller tasks and ensemble them instead of building a monolithic model which sovles everything.

### Problem Statement
Given an Image(I) of a receipt, we want the system to `categorize/summarize the text-content` present in the image.
For now it's safe to assume 
* I is always a scanned image of text document
* I only contains computer generated text i.e. no hand-written content
* Different classes of interest in the content of the receipts are
  * Date of purchase
  * Name (Customer Name)
  * Company (Seller's Name)
  * Receipt number (Some ID: 1270348, or ABC123DEF)
  * Type of good (electronic, food stuff etc.)
  * others (Anything which is doesn't fall into above mentioend categories)


### Pipeline
We have with us all the required information in order to complete the task. Let's hypothesize a pipeline we think will achieve the goal. I am listing these down in chronological order in which the processing will take place by the system but during development we will work on them in a slightly different manner.
1. **Text Detection module**: `Signature: text_detection(Image) -> {(x,y,del_x,, del_y,confidence)}` String is located in the rectangle on the image from pixel(x,y) to pixel(x+del_x,y+del_y). We don't know what the text is but we are very sure(**confidence**) there is some text there. Possible errors the module might generate are:
  a. **False Positives**: The module might and will detect regions with no text as text
  b. **False Negatives**: The module can and most likely will miss detections
  Now is the time to think over the expectations of the module, do we want to detect all the text i.e. reduce the **False Negatives** or do we want to focus and reduce the number of **False Positives**. Ideally, we would like to do both, and the system which achieves it would be our perfect solution. In my approach we will focus on having a **high recall** even if it means we will end up with lots of **false positives**. The idea is that the second module will take care of **false positives**. 
  c. **Level of detection**: Do we want the detections to be at a character level? or word level or sentence level. We will start of with **word level** detections as our aim. 
2. **Text Recognition Module**: `Signature: text_recognition(Image(x,y,del_x,del_y))->{[string, confidence]}` Given a patch of del_x,del_y in the image this module returns the list of most likely words contained in the patch ordered by some confidence.
Again, this module will be not a perfect module and some possible errors it might face are
  a. **Miss-Spelled Words**: A word *burger* can be recognized as a *burgler* or a *bugger*. 
  b. **No Detections**: The algorithm might not even find a word, for cases, when there were no text to begin with. 
