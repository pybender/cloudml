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
  $scope.loadModels = () ->
    # Used for ObjectListCtrl initialization
    (pagination_opts) ->
      Model.$loadAll()

  $scope.test = (model)->
    d = $dialog.dialog(
      modalFade: false
    )
    d.model = model
    d.open('partials/modal.html', 'TestDialogController')
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
        $location.path '/models'
        $scope.$apply()
      ), 300

    ), ((resp) ->
      $scope.saving = false
      $scope.savingError = "Error while saving: server responded with " +
        "#{resp.status} (#{resp.data.error or "no message"}). " +
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
        $location.path '/models'
        $scope.$apply()
      ), 300

    ), ((resp) ->
      $scope.saving = false
      $scope.savingError = "Error while saving: server responded with " +
        "#{resp.status} (#{resp.data.error or "no message"}). " +
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
  if not $scope.model
    if not $routeParams.name
      throw new Error "Can't initialize model detail controller
      without model name"

    $scope.model = new Model({name: $routeParams.name})

  $scope.model.$load().then (->
    $scope.latest_test = new Test($scope.model.latest_test)
    ), (->
      #console.error "Couldn't get model"
      $scope.error = data
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

  $scope.saveImportHandlerChanges = =>
    if not $scope.importHandlerChanged
      return false

    $scope.model.$save(only: ['importhandler']).then (() ->
      $scope.importHandlerChanged = false
    ), (() ->
      throw new Error "Unable to save import handler"
    )

  $scope.$watch 'model.importhandler', (newVal, oldVal) ->
    if newVal? and oldVal? and  newVal != "" and oldVal != ""
      $scope.importHandlerChanged = true

  $scope.test = (model)->
    d = $dialog.dialog(
      modalFade: false
    )
    d.model = model
    d.open('partials/modal.html', 'TestDialogController')
])