'use strict'

### Controllers ###

API_URL = 'http://127.0.0.1:5000/cloudml/b/v1/'

angular.module('app.controllers', ['app.config', ])

.controller('AppCtrl', [
  '$scope'
  '$location'
  '$resource'
  '$rootScope'
  'settings'

($scope, $location, $resource, $rootScope, settings) ->

  # Uses the url to determine if the selected
  # menu item should have the class active.
  $scope.$location = $location
  $scope.$watch('$location.path()', (path) ->
    $scope.activeNavId = path || '/'
  )

  # getClass compares the current url with the id.
  # If the current url starts with the id it returns 'active'
  # otherwise it will return '' an empty string. E.g.
  #
  #   # current url = '/products/1'
  #   getClass('/products') # returns 'active'
  #   getClass('/orders') # returns ''
  #
  $scope.getClass = (id) ->
    if $scope.activeNavId.substring(0, id.length) == id
      return 'active'
    else
      return ''
])

# Controller used for UI Bootstrap pagination
.controller('ObjectListCtrl', [
  '$scope'

  ($scope) ->
    $scope.pages = 0
    $scope.page = 1
    $scope.total = 0
    $scope.per_page = 20

    $scope.objects = []
    $scope.loading = false

    $scope.init = (opts={}) =>
      if not _.isFunction(opts.objectLoader)
        throw new Error "Invalid object loader supplied to ObjectListCtrl"

      $scope.objectLoader = opts.objectLoader
      $scope.load()

      $scope.$watch('page', (page, oldVal, scope) ->
        $scope.load()
      , true)

    $scope.load = =>
      if $scope.loading
        return false
      $scope.loading = true
      $scope.objectLoader(
        page: $scope.page
      ).then ((opts) ->
        $scope.loading = false
        $scope.total = opts.total
        $scope.page = opts.page
        $scope.pages = opts.pages
        $scope.per_page = opts.per_page
        $scope.objects = opts.objects

        # Notify interested parties by emitting and broadcasting an event
        # Event contains
        $scope.$broadcast 'ObjectListCtrl:load:success', $scope.objects
      ), ((opts) ->
        $scope.$broadcast 'ObjectListCtrl:load:error', opts
      )
])