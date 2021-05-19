# SQuAD Subset

This is the folder for SQuAD subset (development set only).

JSON structure & example:
```json
{
  "version": "expmrc-squad-dev",
  "data": [
    {
      "title": "Victoria_(Australia)",
      "paragraphs": [
        {
          "context": "The economy of Victoria is highly diversified: service sectors including financial and property services, health, education, wholesale, retail, hospitality and manufacturing constitute the majority of employment. Victoria's total gross state product (GSP) is ranked second in Australia, although Victoria is ranked fourth in terms of GSP per capita because of its limited mining activity. Culturally, Melbourne is home to a number of museums, art galleries and theatres and is also described as the \"sporting capital of Australia\". The Melbourne Cricket Ground is the largest stadium in Australia, and the host of the 1956 Summer Olympics and the 2006 Commonwealth Games. The ground is also considered the \"spiritual home\" of Australian cricket and Australian rules football, and hosts the grand final of the Australian Football League (AFL) each year, usually drawing crowds of over 95,000 people. Victoria includes eight public universities, with the oldest, the University of Melbourne, having been founded in 1853.",
          "qas": [
            {
              "answers": [
                {
                  "answer_start": 536,
                  "text": "Melbourne"
                },
                {
                  "answer_start": 401,
                  "text": "Melbourne"
                },
                {
                  "answer_start": 401,
                  "text": "Melbourne"
                }
              ],
              "question": "What city in Victoria is called the sporting capital of Australia?",
              "id": "570d2417fed7b91900d45c40",
              "evidences": [
                "Culturally, Melbourne is home to a number of museums, art galleries and theatres and is also described as the \"sporting capital of Australia\".",
                "Culturally, Melbourne is home to a number of museums, art galleries and theatres and is also described as the \"sporting capital of Australia\"."
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

