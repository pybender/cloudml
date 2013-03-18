'use strict'

### Tests specific Controllers ###

angular.module('app.testresults.controllers', ['app.config', ])

.controller('TestDialogController', [
  '$scope'
  '$http'
  'dialog'
  'settings'

($scope, $http, dialog, settings) ->

  model = dialog.model
  $scope.params = model.import_params # list of parameters names
  $scope.parameters = {} # parameters to send via API

  $scope.close = ->
    dialog.close()

  $scope.start = (result) ->
    form_data = new FormData()
    for key of $scope.parameters
      form_data.append(key, $scope.parameters[key])

    $http(
      method: "POST"
      url: settings.apiUrl + "model/#{model.name}/test/test"
      data: form_data
      headers: {'Content-Type':undefined, 'X-Requested-With': null}
      transformRequest: angular.identity
    ).success((data, status, headers, config) ->
      $scope.success = true
      $scope.msg = {}
      dialog.close(result)
    ).error((data, status, headers, config) ->
      $scope.httpError = true
    )
])

.controller('TestDetailsCtrl', [
  '$scope'
  '$http'
  '$routeParams'
  'settings'
  'TestResult'

($scope, $http, $routeParams, settings, Test) ->
  $scope.path = [{label: 'Home', url: '#/'},
                 {label: 'Models', url: '#/models'},
                 {label: 'Model Details', url: ''},
                 {label: 'Test Details', url: ''}]

  if not $scope.test
    if not $routeParams.name
      throw new Error "Can't initialize test detail controller
      without test name"

    $scope.test = new Test({model_name: $routeParams.name,
    name: $routeParams.test_name})

  $scope.test.$load().then (->
    ), (->
      #console.error "Couldn't get test"
      $scope.error = data
      $scope.httpError = true
    )
])