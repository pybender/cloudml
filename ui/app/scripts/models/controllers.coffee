'use strict'

### Trained Model specific Controllers ###

angular.module('app.models.controllers', ['app.config', ])

.controller('ModelListCtrl', [
  '$scope'
  '$http'
  '$dialog'
  'settings'
  'Model'

($scope, $http, $dialog, settings, Model) ->
  Model.$loadAll(
    show: 'name,status,created_on,import_params'
  ).then ((opts) ->
    $scope.objects = opts.objects
  ), ((opts) ->
    $scope.err = "Error while saving: server responded with " +
        "#{resp.status} " +
        "(#{resp.data.response.error.message or "no message"}). " +
        "Make sure you filled the form correctly. " +
        "Please contact support if the error will not go away."
  )
])

.controller('AddModelCtl', [
  '$scope'
  '$http'
  '$location'
  'settings'
  'Model'

($scope, $http, $location, settings, Model) ->
  $scope.model = new Model()
  $scope.new = true

  $scope.upload = ->
    $scope.saving = true
    $scope.savingProgress = '0%'
    $scope.savingError = null

    _.defer ->
      $scope.savingProgress = '50%'
      $scope.$apply()

    $scope.model.$save().then (->
      $scope.savingProgress = '100%'

      _.delay (->
        $location.path $scope.model.objectUrl()
        $scope.$apply()
      ), 300

    ), ((resp) ->
      $scope.saving = false
      $scope.err = "Error while saving: server responded with " +
        "#{resp.status} " +
        "(#{resp.data.response.error.message or "no message"}). " +
        "Make sure you filled the form correctly. " +
        "Please contact support if the error will not go away."
    )

  $scope.setImportHandlerFile = (element) ->
      $scope.$apply ($scope) ->
        $scope.msg = ""
        $scope.error = ""
        $scope.import_handler = element.files[0]
        reader = new FileReader()
        reader.onload = (e) ->
          str = e.target.result
          $scope.model.importhandler = str
          $scope.model.train_importhandler = str
        reader.readAsText($scope.import_handler)

  $scope.setFeaturesFile = (element) ->
      $scope.$apply ($scope) ->
        $scope.msg = ""
        $scope.error = ""
        $scope.features = element.files[0]
        reader = new FileReader()
        reader.onload = (e) ->
          str = e.target.result
          $scope.model.features = str
        reader.readAsText($scope.features)
])

# Upload trained model
.controller('UploadModelCtl', [
  '$scope'
  '$http'
  '$location'
  'settings'
  'Model'

($scope, $http, $location, settings, Model) ->
  $scope.new = true
  $scope.model = new Model()

  $scope.upload = ->
    
    $scope.saving = true
    $scope.savingProgress = '0%'
    $scope.savingError = null
    _.defer ->
      $scope.savingProgress = '50%'
      $scope.$apply()
    $scope.model.$save().then (->
      $scope.savingProgress = '100%'

      _.delay (->
        $location.path $scope.model.objectUrl()
        $scope.$apply()
      ), 300

    ), ((resp) ->
      $scope.saving = false
      $scope.err = "Error while saving: server responded with " +
        "#{resp.status} " +
        "(#{resp.data.response.error.message or "no message"}). " +
        "Make sure you filled the form correctly. " +
        "Please contact support if the error will not go away."
    )

  $scope.setModelFile = (element) ->
      $scope.$apply ($scope) ->
        $scope.msg = ""
        $scope.error = ""
        $scope.model_file = element.files[0]
        $scope.model.trainer = element.files[0]

  $scope.setImportHandlerFile = (element) ->
      $scope.$apply ($scope) ->
        $scope.msg = ""
        $scope.error = ""
        $scope.import_handler = element.files[0]
        reader = new FileReader()
        reader.onload = (e) ->
          str = e.target.result
          $scope.model.importhandler = str
        reader.readAsText($scope.import_handler)
])

.controller('ModelDetailsCtrl', [
  '$scope'
  '$http'
  '$location'
  '$routeParams'
  '$dialog'
  'settings'
  'Model'
  'TestResult'

($scope, $http, $location, $routeParams, $dialog, settings, Model, Test) ->
  $scope.msg = ''

  if not $scope.model
    if not $routeParams.name
      throw new Error "Can't initialize model detail controller
      without model name"

    $scope.model = new Model({name: $routeParams.name})

  $scope.model.$load().then (->
    $scope.latest_test = new Test($scope.model.latest_test)
    ), (->
      #console.error "Couldn't get model"
      $scope.err = data
      $scope.httpError = true
    )

  DEFAULT_ACTION = 'model:details'
  $scope.action = ($routeParams.action or DEFAULT_ACTION).split ':'
  $scope.$watch 'action', (action) ->
    actionString = action.join(':')
    $location.search(
      if actionString == DEFAULT_ACTION then ""
      else "action=#{actionString}")

  $scope.toggleAction = (action) =>
    $scope.action = action

  $scope.loadTests = () ->
    (pagination_opts) ->
      Test.$loadTests($scope.model.name)

  $scope.saveTrainHandler = =>
    $scope.model.$save(only: ['train_importhandler']).then (() ->
      $scope.msg = 'Import Handler for training model saved'
    ), (() ->
      throw new Error "Unable to save import handler"
    )

  $scope.saveTestHandler = =>
    $scope.model.$save(only: ['importhandler']).then (() ->
      $scope.msg = 'Import Handler for tests saved'
    ), (() ->
      throw new Error "Unable to save import handler"
    )
])

.controller('TrainModelCtrl', [
  '$scope'
  '$http'
  'dialog'
  'settings'

  ($scope, $http, dialog, settings) ->

    $scope.model = dialog.model
    $scope.params = $scope.model.import_params
    $scope.parameters = {}

    $scope.close = ->
      dialog.close()

    $scope.start = (result) ->
      $scope.model.$train($scope.parameters).then (() ->
        $scope.close()
      ), (() ->
        throw new Error "Unable to start model training"
      )
])

.controller('ModelActionsCtrl', [
  '$scope'
  '$dialog'

  ($scope, $dialog) ->
    $scope.init = (opts={}) =>
      if not opts.model
        throw new Error "Please specify model"

      $scope.model = opts.model

    $scope.test_model = (model)->
      d = $dialog.dialog(
        modalFade: false
      )
      d.model = model
      d.open('partials/modal.html', 'TestDialogController')

    $scope.train_model = (model)->
      d = $dialog.dialog(
        modalFade: false
      )
      d.model = model
      d.open('partials/model_train_popup.html', 'TrainModelCtrl')
  
])
