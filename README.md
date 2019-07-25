PyPastry - the opinionated machine learning experimentation framework
=====================================================================

PyPastry is a framework for developers and data scientists to run
machine learning experiments. We enable you to:

 - Iterate quickly. The more experiments you do, the more likely you
   are to find something that works well.
 - Experiment correctly and consistently. Anything else is not really
   an experiment, is it?
 - Make experiments reproducible. That means keeping track of your
   code state and results.
 - Experiment locally. None of that Spark rubbish.
 - Use standard tools. Everything is based on Scikit-learn and Git.

Quick start
-----------

    > conda install pastry
	> mkdir pastry-test; cd pastry-test
	> pastry init
    Created new project in ~/pastry-test
    > pastry -m "First experiment"
	Running 1 experiment...

          Date  Git hash   Dataset  Predictor  F1 score  F1 score SEM  Duration (s)           Comment
    2019-07-01   98350bb  iris.csv  LinearSVC     0.153         0.052          0.32  First experiment
