Output Viewer
=============

We include a small application to view run outputs in a web browser. See the
[README](../README.md) for basic installation instructions.

Application structure
---------------------

```
app/
├── frontend                SolidJS web application 
└── scripts                 Data preparation scripts 
```

Components
----------

+ **RunOverview** - Lists available runs and displays metadata, accuracy
   timelines
+ **StepViewer** - Navigate through ablation steps, view performance metrics
+ **QuestionViewer** - Browse questions/responses with filtering and search
+ **NgramAnalysis** - Explore ablated n-grams and distinctiveness scores

