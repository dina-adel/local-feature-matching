# Local Feature Matching
The objective of this project is to learn about extracting features from images and matching them. <br>
The project's logic is implemented in `student.py` in which I implemented three functions: <br>
- `get_interest_points`: extracts interest points using Harris Corner Detector.
- `get_features`: generates SIFT-like features. 
- `match_features`: matches the features generated from two images.

**To test the project, run**:`python main.py` then, you will be prompted to input a file name. 
Enter one of the following: `notre_dame`, `mt_rushmore` or `e_gaudi`. <br>

**Results**
---------- 

- **Notre Dame** <br>
![notre dame results](./results/notre_dame_matches.jpg)
> Number of matches after running the pipeline: 1102 <br>
> Accuracy on 50 most confident: 100% <br>
> Accuracy on 100 most confident: 98% <br> 
> Accuracy on all matches: 74% <br>

- **Mt Rushmore** <br>
![mt rushmore results](./results/mt_rushmore_matches.jpg) 
> Number of matches after running the pipeline: 170 <br>
> Accuracy on 50 most confident: 96% <br>
> Accuracy on 100 most confident: 94% <br>
> Accuracy on all matches: 91% <br>

- **E Gaudi** <br>
![E Gaudi results](./results/e_gaudi_matches.jpg) 
> Number of matches after running the pipeline: 44 <br>
> Accuracy on all matches: 6% <br>
