angular.module('app.models.model', ['app.config'])

.factory('Model', [
  '$http'
  '$q'
  'settings'
  
  ($http, $q, settings) ->

    trimTrailingWhitespace = (val) -> val.replace /^\s+|\s+$/g, ''

    ###
    Trained Model
    ###
    class Model

      constructor: (opts) ->
        @loadFromJSON opts

      id: null
      # Unix time of model creation
      created_on: null
      status: null
      name: null
      trainer: null
      importParams: null
      negative_weights: null
      negative_weights_tree: null
      positive_weights: null
      positive_weights_tree: null
      latest_test: null
      importhandler: null
      train_importhandler: null
      features: null

      ### API methods ###

      isNew: -> if @id == null then true else false

      # Returns an object of job properties, for use in e.g. API requests
      # and templates
      toJSON: =>
        importhandler: @importhandler
        trainer: @trainer
        features: @features

      # Sets attributes from object received e.g. from API response
      loadFromJSON: (origData) =>
        data = _.extend {}, origData
        _.extend @, data

      $load: =>
        if @name == null
          throw new Error "Can't load model without name"

        $http(
          method: 'GET'
          url: settings.apiUrl + "model/#{@name}"
          headers:
            'X-Requested-With': null
        ).then ((resp) =>
          @loaded = true
          @loadFromJSON(resp.data['model'])
          return resp

        ), ((resp) =>
          return resp
        )


      prepareSaveJSON: (json) =>
        reqData = json or @toJSON()
        return reqData

      # Makes PUT or POST request to save the object. Options:
      # ``only``: may contain a list of fields that will be sent to the server
      # (only when PUTting to existing objects, API allows partial update)
      $save: (opts={}) =>
        #saveData = @toJSON()
        fd = new FormData()
        fd.append("trainer", @trainer)
        fd.append("importhandler", @importhandler)
        fd.append("train_importhandler", @train_importhandler)
        fd.append("features", @features)

        # fields = opts.only || []
        # if fields.length > 0
        #   for key in _.keys(saveData)
        #     if key not in fields
        #       delete saveData[key]

        #saveData = @prepareSaveJSON(saveData)
        $http(
          method: if @isNew() then "POST" else "PUT"
          #headers: settings.apiRequestDefaultHeaders
          headers: {'Content-Type':undefined, 'X-Requested-With': null}
          url: "#{settings.apiUrl}model/#{@name or ""}"
          data: fd
          transformRequest: angular.identity
        )
        .then((resp) => @loadFromJSON(resp.data['model']))

      # Requests all available jobs from API and return a list of
      # Job instances
      @$loadAll: (opts) ->
        dfd = $q.defer()

        $http(
          method: 'GET'
          url: "#{settings.apiUrl}model/"
          headers: settings.apiRequestDefaultHeaders
        )
        .then ((resp) =>
          dfd.resolve {
            total: resp.data.found
            objects: (
              new @(_.extend(obj, {loaded: true})) \
              for obj in resp.data.models)
            _resp: resp
          }

        ), (-> dfd.reject.apply @, arguments)

        dfd.promise

      $train: (opts={}) =>
        fd = new FormData()
        for key, val of opts
          fd.append(key, val)
        
        $http(
          method: "PUT"
          #headers: settings.apiRequestDefaultHeaders
          headers: {'Content-Type':undefined, 'X-Requested-With': null}
          url: "#{settings.apiUrl}model/#{@name}/train"
          data: fd
          transformRequest: angular.identity
        )
        .then((resp) => @loadFromJSON(resp.data['model']))

    return Model
])