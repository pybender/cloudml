angular.module('app.testresults.model', ['app.config'])

.factory('TestResult', [
  '$http'
  '$q'
  'settings'
  'Model'
  
  ($http, $q, settings, Model) ->

    trimTrailingWhitespace = (val) -> val.replace /^\s+|\s+$/g, ''

    ###
    Trained Model
    ###
    class TestResult

      constructor: (opts) ->
        @loadFromJSON opts

      id: null
      accuracy: null
      created_on: null
      data_count: null
      name: null
      parameters: null
      model: null
      model_name: null

      ### API methods ###

      isNew: -> if @slug == null then true else false

      # Returns an object of job properties, for use in e.g. API requests
      # and templates
      toJSON: =>
        name: @name

      # Sets attributes from object received e.g. from API response
      loadFromJSON: (origData) =>
        data = _.extend {}, origData
        _.extend @, data

      $load: =>
        if @name == null
          throw new Error "Can't load model without name"
        $http(
          method: 'GET'
          url: settings.apiUrl + "model/#{@model_name}/test/#{@name}"
          headers: {'X-Requested-With': null}
        ).then ((resp) =>
          @loaded = true
          @loadFromJSON(resp.data['test'])
          @model = new Model(resp.data['model'])
          return resp

        ), ((resp) =>
          return resp
        )

      # Makes PUT or POST request to save the object. Options:
      # ``only``: may contain a list of fields that will be sent to the server
      # (only when PUTting to existing objects, API allows partial update)
      $save: (opts={}) =>
        saveData = @toJSON()

        fields = opts.only || []
        if fields.length > 0
          for key in _.keys(saveData)
            if key not in fields
              delete saveData[key]

        saveData = @prepareSaveJSON(saveData)

        $http(
          method: if @isNew() then 'POST' else 'PUT'
          headers: settings.apiRequestDefaultHeaders
          url: "#{settings.apiUrl}/jobs/#{@id or ""}"
          params: {access_token: user.access_token}
          data: $.param saveData
        )
        .then((resp) => @loadFromJSON(resp.data))

      
      @$loadTests: (modelName, opts) ->
        dfd = $q.defer()

        if not modelName then throw new Error "Model is required to load tests"

        $http(
          method: 'GET'
          url: "#{settings.apiUrl}model/#{modelName}/tests"
          headers: settings.apiRequestDefaultHeaders
          params: _.extend {
          }, opts
        )
        .then ((resp) =>
          dfd.resolve {
            total: resp.data.found
            objects: (new @(obj) for obj in resp.data.tests)
            _resp: resp
          }

        ), (-> dfd.reject.apply @, arguments)

        dfd.promise
         
    return TestResult
])