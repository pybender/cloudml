'use strict'

### Controllers ###

API_URL = 'http://127.0.0.1:5000/cloudml/b/v1/'

angular.module('app.controllers', ['app.config', ])


.controller('ObjectListCtrl', [
  '$scope'

  ($scope) ->
    $scope.objNumTotal = 0
    $scope.objNumDisplayed = 0
    $scope.objects = []
    $scope.objPerLoad = 10
    $scope.haveMoreToLoad = true
    $scope.loadingMore = false

    $scope.init = (opts={}) =>
      if not _.isFunction(opts.objectLoader)
        throw new Error "Invalid object loader supplied to ObjectListCtrl"

      $scope.objectLoader = opts.objectLoader
      $scope.loadMore()

    $scope.loadMore = =>
      if $scope.loadingMore
        return false

      $scope.loadingMore = true

      $scope.objectLoader(
        count: $scope.objPerLoad
        offset: $scope.objNumDisplayed
      ).then ((opts) ->
        $scope.loadingMore = false

        $scope.objNumTotal = opts.total
        $scope.objects.push.apply $scope.objects, opts.objects

        objNumDisplayedBeforeUpdate = $scope.objNumDisplayed
        $scope.objNumDisplayed = $scope.objects.length
        if $scope.objNumDisplayed == objNumDisplayedBeforeUpdate
          $scope.haveMoreToLoad = false

        # Notify interested parties by emitting and broadcasting an event
        # Event contains
        $scope.$broadcast 'ObjectListCtrl:load:success', $scope.objects

      ), ((opts) ->
        $scope.$broadcast 'ObjectListCtrl:load:error', opts

      )
])

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

.controller('Model_list', [
  '$scope'
  '$http'
  '$dialog'
  'settings'
  'Model'

($scope, $http, $dialog, settings, Model) ->
  $scope.path = [{label: 'Home', url: '#/'}, {label: 'Models', url: '#/models'}]
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

($scope, $http, $location, settings) ->
  $scope.path = [{label: 'Home', url: '#/'},
  {label: 'Train Model', url: '#/add_model'}]
  $scope.upload = ->
    fd = new FormData()
    fd.append("import_handler_local", $scope.import_handler_local)
    fd.append("features", $scope.features)
    $http(
      method: "POST"
      url: settings.apiUrl + "model/train/#{$scope.name}"
      data: fd
      headers: {'Content-Type':undefined, 'X-Requested-With': null}
      transformRequest: angular.identity
    ).success((data, status, headers, config) ->
      $scope.msg = data.name
      $location.path '/models'
    )
  $scope.setImportHandlerLocalFile = (element) ->
      $scope.$apply ($scope) ->
        $scope.msg = ""
        $scope.error = ""
        $scope.import_handler_local = element.files[0]

  $scope.setFeaturesFile = (element) ->
      $scope.$apply ($scope) ->
        $scope.msg = ""
        $scope.error = ""
        $scope.features = element.files[0]
])

.controller('UploadModelCtl', [
  '$scope'
  '$http'
  '$location'
  'settings'

($scope, $http, $location, settings) ->
  $scope.path = [{label: 'Home', url: '#/'},
  {label: 'Upload Trained Model', url: '#/upload_model'}]
  $scope.upload = ->
    fd = new FormData()
    fd.append("file", $scope.file)
    fd.append("import_handler_local", $scope.import_handler_local)
    fd.append("features", $scope.features)
    $http(
      method: "POST"
      url: settings.apiUrl + "model/#{$scope.name}"
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

  $scope.setFeaturesFile = (element) ->
      $scope.$apply ($scope) ->
        $scope.msg = ""
        $scope.error = ""
        $scope.features = element.files[0]
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
  $scope.path = [{label: 'Home', url: '#/'},
                 {label: 'Models', url: '#/models'},
                 {label: 'Model Details', url: ''}]

  if not $scope.model
    if not $routeParams.name
      throw new Error "Can't initialize model detail controller
      without model name"

    $scope.model = new Model({name: $routeParams.name})

  $scope.model.$load().then (->
    $scope.latest_test = new Test($scope.model.latest_test)
    ), (->
      console.error "Couldn't get model"
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

    $scope.model.$save(only: ['import_handler']).then (() ->
      $scope.importHandlerChanged = false
    ), (() ->
      throw new Error "Unable to save import handler"
    )

  $scope.$watch 'model.import_handler', (newVal, oldVal) ->
    if newVal? and oldVal? and  newVal != "" and oldVal != ""
      $scope.importHandlerChanged = true

  $scope.test = (model)->
    d = $dialog.dialog(
      modalFade: false
    )
    d.model = model
    d.open('partials/modal.html', 'TestDialogController')
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
      console.error "Couldn't get test"
      $scope.error = data
      $scope.httpError = true
    )

])

.controller('TestExamplesCtrl', [
  '$scope'
  '$http'
  '$routeParams'
  'settings'

($scope, $http, $routeParams, settings) ->
  $scope.path = [{label: 'Home', url: '#/'},
                 {label: 'Models', url: '#/models'},
                 {label: 'Model Details', url: "#/models/#{$routeParams.name}"},
                 {label: 'Test Details', url: "#/models/#{$routeParams.name}/\
tests/#{$routeParams.test_name}"},
                 {label: 'Test Examples', url: ''}]
  $scope.currentPage = 1
  $scope.$watch('currentPage',
    (currentPage, oldVal, scope) ->
      $http(
        method: 'GET'
        url: settings.apiUrl +
          "model/#{$routeParams.name}/test/#{$routeParams.test_name}\
/data?page=#{currentPage}"
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
  'settings'

($scope, $http, $routeParams, settings) ->
  $scope.path = [{label: 'Home', url: '#/'},
                 {label: 'Models', url: '#/models'},
                 {label: 'Model Details', url: ''},
                 {label: 'Test Details', url: ''},
                 {label: 'Test Examples', url: ''},
                 {label: 'Example Details', url: ''}]
  $http(
    method: 'GET'
    url: settings.apiUrl +
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
