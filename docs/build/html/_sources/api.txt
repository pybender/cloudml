===
Api
===


Models
======

+--------+-------------------------------------------------------+--------------------+
| Method | Resource                                              | Description        |
+========+=======================================================+====================+
| GET    | :ref:`/cloudml/b/v1/model/<list_models>`              | Get list of models |
+--------+-------------------------------------------------------+--------------------+
| GET    | :ref:`/cloudml/b/v1/model/:ModelName<get_model>`      | Get model by name  |
+--------+-------------------------------------------------------+--------------------+
| POST   | :ref:`/cloudml/b/v1/model/:ModelName/predict<predict>`| Predict            |
+--------+-------------------------------------------------------+--------------------+

.. _list_models:

List of Models
--------------

* Method: GET
* URL: /cloudml/b/v1/model/

Response Parameters
^^^^^^^^^^^^^^^^^^^

* models - List of models


Errors
^^^^^^

404 Not Found::

    {
      "response": {
        "server_time": 1364714381.157842, 
        "error": {
          "status": 404, 
          "debug": null, 
          "message": "Model doesn't exist", 
          "code": 1001
        }
      }
    }

Example
^^^^^^^

cURL::

    curl http://127.0.0.1:5000/cloudml/b/v1/model/

Response body::
    
    {
        "models": [
            {
                "name": "test",
                "status": "Trained",
                "created_on": "2013-03-28T05:47:46.566463"
                ...
            }
        ]
    }
    


.. _get_model:

Get Model
---------

* Method: GET
* URL: /cloudml/b/v1/model/:ModelName

Request paraneters
^^^^^^^^^^^^^^^^^^

* ModelName - name of model


Response Parameters
^^^^^^^^^^^^^^^^^^^

* id - internal id of model
* name - name of model
* status - status of model [New, Trained]
* created_on - date and tiome of create
* importhandler - importhandler for testing model
* target_variable - target variable

Errors
^^^^^^

404 Not Found::

    {
      "response": {
        "server_time": 1364714381.157842, 
        "error": {
          "status": 404, 
          "debug": null, 
          "message": "Model doesn't exist", 
          "code": 1001
        }
      }
    }

Example
^^^^^^^

cURL::

    curl http://127.0.0.1:5000/cloudml/b/v1/model/test

Response body::
    
    {
        "model": {
            "name": "test",
            "status": "Trained",
            "created_on": "2013-03-28T05:47:46.566463"
            ...
        }
    }


.. _predict:

Predict
-------

* Method: POST
* URL: /cloudml/b/v1/model/:ModelName/predict

Request paraneters
^^^^^^^^^^^^^^^^^^

* ModelName - name of model

Response Parameters
^^^^^^^^^^^^^^^^^^^

Errors
^^^^^^
400 Bad Request::

    {
      "response": {
        "server_time": 1364714887.802514, 
        "error": {
          "status": 400, 
          "debug": null, 
          "message": "400 Bad Request", 
          "code": 1005
        }
      }
    }

404 Not Found::

    {
      "response": {
        "server_time": 1364714381.157842, 
        "error": {
          "status": 404, 
          "debug": null, 
          "message": "Model doesn't exist", 
          "code": 1001
        }
      }
    }

Example
^^^^^^^

cURL::

    curl -XPOST http://127.0.0.1:5000/cloudml/b/v1/model/test/predict H "Accept: application/json" -H "Content-type: application/json" -d @request.json

Request body::

    [
        {
            "application": "aplication",
            "opening": "opening",
            "hire_outcome": "1",
            "contractor_info":{
                "country": "Ukrain",
                "dev_timezone": "dev_timezone"
                ....
            },
            "employer_info": {
                "country": "Ukrain",
                "op_timezone": "GMT2",
                "op_country_tz": "GMT1"
                ...
            }
            ...
        },
        ...
]

Response body::
    
    [
        {
            "item": 0, 
            "probs": [
                0.5322514763217836, 
                0.46774852367821645
            ], 
            "label": [
                1,
                0
            ]
        }, 
        ...
    ]

