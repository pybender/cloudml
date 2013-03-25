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
      test_name: $routeParams.test_name,
      show:'id,label,pred_label,title'}, pagination_opts))
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

  $scope.data.$load(
    show: "id,weighted_data_input,target_variable,pred_label,label"
  ).then (->
    ), (->
      $scope.error = data
      $scope.httpError = true
    )
])