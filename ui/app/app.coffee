'use strict'

# Declare app level module which depends on filters, and services
App = angular.module('app', [
  'ui'
  'ngCookies'
  'ngResource'
  'app.controllers'
  'app.directives'
  'app.filters'
  'app.services'
  'ui.bootstrap'
])
App.config([
  '$routeProvider'
  '$locationProvider'

($routeProvider, $locationProvider, config) ->

  $routeProvider

    .when('/models', {
      templateUrl: '/partials/model_list.html'
      controller: 'Model_list'
    })
    .when('/models/:name', {
      controller: 'ModelDetailsCtrl'
      templateUrl: '/partials/model_details.html'
    })
    .when('/models/:name/tests/:test_name', {
      controller: 'TestDetailsCtrl'
      templateUrl: '/partials/test_details.html'
    })
    .when('/models/:name/tests/:test_name/examples', {
      controller: 'TestExamplesCtrl'
      templateUrl: '/partials/test_examples.html'
    })
    .when('/models/:name/tests/:test_name/example/:data_id', {
      controller: 'ExampleDetailsCtrl'
      templateUrl: '/partials/example_details.html'
    })
    .when('/upload_model', {
      templateUrl: '/partials/upload_model.html'
      controller: 'UploadModelCtl'
    })
    .when('/add_model', {
      templateUrl: '/partials/add_model.html'
      controller: 'AddModelCtl'
    })

    # Catch all
    .otherwise({redirectTo: '/models'})

  # Without server side support html5 must be disabled.
  $locationProvider.html5Mode(false)
])

App.run(['$rootScope', ($rootScope) ->
  $rootScope.Math = window.Math

  # this will be available to all scope variables
  $rootScope.includeLibraries = true

  # this method will be available to all scope variables as well
  $rootScope.include = (libraries) ->
    scope = this
    # attach each of the libraries directly to the scope variable
    for key of libraries
      scope[key] = getLibrary(key)
    return scope

])