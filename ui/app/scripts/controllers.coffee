'use strict'

### Controllers ###

API_URL = 'http://127.0.0.1:5000/cloudml/b/v1/'

angular.module('app.controllers', [])

.controller('AppCtrl', [
  '$scope'
  '$location'
  '$resource'
  '$rootScope'

($scope, $location, $resource, $rootScope) ->

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

.controller('TestDialogController', [
  '$scope'
  '$http'
  'dialog'

($scope, $http, dialog) ->

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
      url: "http://127.0.0.1:5000/cloudml/b/v1/model/#{model.name}/test/test"
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

.controller('Model_list', [
  '$scope'
  '$http'
  '$dialog'

($scope, $http, $dialog) ->
  $scope.path = [{label: 'Home', url: '#/'}, {label: 'Models', url: '#/models'}]
  $http(
    method: 'GET'
    url: "http://127.0.0.1:5000/cloudml/b/v1/model"
    headers: {'X-Requested-With': null}
  ).success((data, status, headers, config) ->
      $scope.models = data.models
  ).error((data, status, headers, config) ->
      $scope.error = data
  )
  $scope.test = (model)->
    d = $dialog.dialog(
      modalFade: false
    )
    d.model = model
    d.open('partials/modal.html', 'TestDialogController')
])

.controller('UploadModelCtl', [
  '$scope'
  '$http'
  '$location'

($scope, $http, $location) ->
  $scope.path = [{label: 'Home', url: '#/'},
  {label: 'Upload Trained Model', url: '#/upload_model'}]
  $scope.upload = ->
    fd = new FormData()
    fd.append("file", $scope.file)
    fd.append("import_handler_local", $scope.import_handler_local)
    $http(
      method: "POST"
      url: "http://127.0.0.1:5000/cloudml/b/v1/model/#{$scope.name}"
      data: fd
      headers: {'Content-Type':undefined, 'X-Requested-With': null}
      transformRequest: angular.identity
    ).success((data, status, headers, config) ->
      $scope.msg = data.name
      $location.path '/models'
    )
  $scope.setFile = (element) ->
      $scope.$apply ($scope) ->
        $scope.msg = ""
        $scope.error = ""
        $scope.file = element.files[0]

  $scope.setImportHandlerLocalFile = (element) ->
      $scope.$apply ($scope) ->
        $scope.msg = ""
        $scope.error = ""
        $scope.import_handler_local = element.files[0]
])

.controller('ModelDetailsCtrl', [
  '$scope'
  '$http'
  '$routeParams'

($scope, $http, $routeParams) ->
  $scope.path = [{label: 'Home', url: '#/'},
                 {label: 'Models', url: '#/models'},
                 {label: 'Model Details', url: ''}]
  $http(
    method: 'GET'
    url: "http://127.0.0.1:5000/cloudml/b/v1/model/#{$routeParams.name}"
    headers: {'X-Requested-With': null}
  ).success((data, status, headers, config) ->
      $scope.model = data.model
      $scope.tests = data.tests
  ).error((data, status, headers, config) ->
      $scope.error = data
  )
])

.controller('TestDetailsCtrl', [
  '$scope'
  '$http'
  '$routeParams'

($scope, $http, $routeParams) ->
  $scope.path = [{label: 'Home', url: '#/'},
                 {label: 'Models', url: '#/models'},
                 {label: 'Model Details', url: ''},
                 {label: 'Test Details', url: ''}]
  $http(
    method: 'GET'
    url: API_URL + "model/#{$routeParams.name}/test/#{$routeParams.test_name}"
    headers: {'X-Requested-With': null}
  ).success((data, status, headers, config) ->
      $scope.test = data.test
      $scope.metrics = data.metrics
      $scope.model = data.model
  ).error((data, status, headers, config) ->
      $scope.error = data
  )

])

.controller('TestExamplesCtrl', [
  '$scope'
  '$http'
  '$routeParams'

($scope, $http, $routeParams) ->
  $scope.path = [{label: 'Home', url: '#/'},
                 {label: 'Models', url: '#/models'},
                 {label: 'Model Details', url: ''},
                 {label: 'Test Details', url: ''},
                 {label: 'Test Examples', url: ''}]
  $scope.currentPage = 1
  $scope.$watch('data.page',
    (currentPage, oldVal, scope) ->
      if currentPage
        alert(currentPage)
      $http(
        method: 'GET'
        url: API_URL +
          "model/#{$routeParams.name}/test/#{$routeParams.test_name}/data"
        headers: {'X-Requested-With': null}
      ).success((data, status, headers, config) ->
          $scope.data = data.data
          $scope.test = data.test
          $scope.model = data.model
      ).error((data, status, headers, config) ->
          $scope.error = data
      )
    , true)

])

.controller('ExampleDetailsCtrl', [
  '$scope'
  '$http'
  '$routeParams'

($scope, $http, $routeParams) ->
  $scope.path = [{label: 'Home', url: '#/'},
                 {label: 'Models', url: '#/models'},
                 {label: 'Model Details', url: ''},
                 {label: 'Test Details', url: ''},
                 {label: 'Test Examples', url: ''},
                 {label: 'Example Details', url: ''}]
  $http(
    method: 'GET'
    url: API_URL +
      "model/#{$routeParams.name}/test/#{$routeParams.test_name}/" +
      "data/#{$routeParams.data_id}"
    headers: {'X-Requested-With': null}
  ).success((data, status, headers, config) ->
      $scope.data = data.data
      $scope.test = data.test
      $scope.model = data.model

  ).error((data, status, headers, config) ->
      $scope.error = data
  )

])
