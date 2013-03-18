'use strict'

### Controllers ###

API_URL = 'http://127.0.0.1:5000/cloudml/b/v1/'

angular.module('app.controllers', ['app.config', ])


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
  'Model'

($scope, $http, $location, settings, Model) ->
  $scope.path = [{label: 'Home', url: '#/'},
  {label: 'Add Model', url: '#/add_model'}]

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
  $scope.path = [{label: 'Home', url: '#/'},
  {label: 'Upload Trained Model', url: '#/upload_model'}]
 
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
