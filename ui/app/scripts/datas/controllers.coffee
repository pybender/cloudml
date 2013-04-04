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

.controller('GroupedExamplesCtrl', [
  '$scope'
  '$http'
  '$routeParams'
  'settings'
  'Data'

($scope, $http, $routeParams, settings, Data) ->
  $scope.test_name = $routeParams.test_name
  $scope.model_name = $routeParams.name
  Data.$loadAllGroupped(
    model_name: $routeParams.name
    test_name: $routeParams.test_name
  ).then ((opts) ->
    $scope.field_name = opts.field_name
    $scope.mavp = opts.mavp
    $scope.objects = opts.objects
  ), ((opts) ->
    $scope.err = "Error while loading: server responded with " +
        "#{opts.status} " +
        "(#{opts.data.response.error.message or "no message"})."
  )
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