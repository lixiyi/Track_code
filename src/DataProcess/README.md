# Data Observation
## Three rules:
* No wire service articles.  (That is, from Associated Press (AP), AFP, etc)
    * judge wire service articles as not relevant​
* No opinion or editorials.
    *  "Opinion", "Letters to the Editor", or "The Post's View" sections, as labeled in the "kicker" field, are​not relevant​.
* The list of links should be diverse.
    * not sure, waiting...


## Calculate
#### Level 1:
* fields:
    * id 595037
    * article_url 595037
    * title 595037
    * author 595037
    * published_date 595037
    * contents 595037
    * type 595037
    * source 595037
* type:
    * article 236649
    * blog 358388

#### Level 2:(Contents)
* About 414 examples contain 'null' element


## Processing
* Skip 'null' elements in Contents field
* Remove opinion or editorials file according to the "kicker" field.