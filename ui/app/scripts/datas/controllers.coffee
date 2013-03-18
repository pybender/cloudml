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