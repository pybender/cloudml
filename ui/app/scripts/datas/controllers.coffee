'use strict'

### Tests examples specific Controllers ###

angular.module('app.datas.controllers', ['app.config', ])

.controller('TestExamplesCtrl', [
  '$scope'
  '$http'
  '$routeParams'
  'settings'
  'Data'

($scope, $http, $routeParams, settings, Data) ->
  $scope.path = [{label: 'Home', url: '#/'},
                 {label: 'Models', url: '#/models'},
                 {label: 'Model Details', url: "#/models/#{$routeParams.name}"},
                 {label: 'Test Details', url: "#/models/#{$routeParams.name}/\
tests/#{$routeParams.test_name}"},
                 {label: 'Test Examples', url: ''}]
  $scope.test_name = $routeParams.test_name
  $scope.loadDatas = () ->
    # Used for ObjectListCtrl initialization
    (pagination_opts) ->
      Data.$loadAll(_.extend({model_name: $routeParams.name,
      test_name: $routeParams.test_name}, pagination_opts))
])

.controller('ExampleDetailsCtrl', [
  '$scope'
  '$http'
  '$routeParams'
  'settings'
  'Data'

($scope, $http, $routeParams, settings, Data) ->
  $scope.path = [{label: 'Home', url: '#/'},
                 {label: 'Models', url: '#/models'},
                 {label: 'Model Details', url: ''},
                 {label: 'Test Details', url: ''},
                 {label: 'Test Examples', url: ''},
                 {label: 'Example Details', url: ''}]
  if not $scope.data
    $scope.data = new Data({model_name: $routeParams.name,
    test_name: $routeParams.test_name,
    id: $routeParams.data_id})

  $scope.data.$load().then (->
    ), (->
      $scope.error = data
      $scope.httpError = true
    )
])