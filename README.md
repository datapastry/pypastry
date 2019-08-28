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
 - Use standard tools. Everything is based on Scikit-learn, Pandas and Git.

Quick start
-----------

    > git clone https://github.com/datapastry/pypastry.git
    > pip install -e pypastry
	> pastry init pastry-test
    > cd pastry-test
    > pastry run -m "First experiment"
    Got dataset with 10 rows
       Git hash Dataset hash            Run start                   Model          Score Duration (s)
    0  aa87ce62     71e8f4fd  2019-08-28 06:39:07  DecisionTreeClassifier  0.933 ± 0.067         0.03

Contributing
------------

PyPastry is at an early stage so there's plenty to do and we'd love to have your contribution.

Check out the issues for a list of things that need doing and post a comment if you'd like to take
something on.

If you have an idea for something you'd like to do, create an issue.

Thanks for using PyPastry!
