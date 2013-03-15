'use strict';

var App;

App = angular.module('app', ['ui', 'ngCookies', 'ngResource', 'app.config', 'app.controllers', 'app.directives', 'app.filters', 'app.services', 'ui.bootstrap']);

App.config([
  '$routeProvider', '$locationProvider', function($routeProvider, $locationProvider, config) {
    $routeProvider.when('/models', {
      templateUrl: '/partials/model_list.html',
      controller: 'Model_list'
    }).when('/models/:name', {
      controller: 'ModelDetailsCtrl',
      templateUrl: '/partials/model_details.html'
    }).when('/models/:name/tests/:test_name', {
      controller: 'TestDetailsCtrl',
      templateUrl: '/partials/test_details.html'
    }).when('/models/:name/tests/:test_name/examples', {
      controller: 'TestExamplesCtrl',
      templateUrl: '/partials/test_examples.html'
    }).when('/models/:name/tests/:test_name/example/:data_id', {
      controller: 'ExampleDetailsCtrl',
      templateUrl: '/partials/example_details.html'
    }).when('/upload_model', {
      templateUrl: '/partials/upload_model.html',
      controller: 'UploadModelCtl'
    }).when('/add_model', {
      templateUrl: '/partials/add_model.html',
      controller: 'AddModelCtl'
    }).otherwise({
      redirectTo: '/models'
    });
    return $locationProvider.html5Mode(false);
  }
]);

App.run([
  '$rootScope', function($rootScope) {
    $rootScope.Math = window.Math;
    $rootScope.includeLibraries = true;
    return $rootScope.include = function(libraries) {
      var key, scope;
      scope = this;
      for (key in libraries) {
        scope[key] = getLibrary(key);
      }
      return scope;
    };
  }
]);
'use strict';

/* Configuration
*/

var SETTINGS;

SETTINGS = {
  apiUrl: 'http://127.0.0.1:5000/cloudml/b/v1/',
  apiRequestDefaultHeaders: {
    'Content-Type': 'application/x-www-form-urlencoded',
    'X-Requested-With': null
  }
};

angular.module('app.config', []).config([
  '$provide', function($provide) {
    var local_settings, local_settings_injector;
    try {
      angular.module('app.local_config');
    } catch (e) {
      console.warn("Couldn't find local settings: " + e.message);
      angular.module('app.local_config', []).constant('settings', {});
    }
    local_settings_injector = angular.injector(['app.local_config']);
    local_settings = local_settings_injector.get('settings');
    return $provide.constant('settings', $.extend(SETTINGS, local_settings));
  }
]);
'use strict';

/* Controllers
*/

var API_URL;

API_URL = 'http://127.0.0.1:5000/cloudml/b/v1/';

angular.module('app.controllers', ['app.config']).controller('AppCtrl', [
  '$scope', '$location', '$resource', '$rootScope', 'settings', function($scope, $location, $resource, $rootScope, settings) {
    $scope.$location = $location;
    $scope.$watch('$location.path()', function(path) {
      return $scope.activeNavId = path || '/';
    });
    return $scope.getClass = function(id) {
      if ($scope.activeNavId.substring(0, id.length) === id) {
        return 'active';
      } else {
        return '';
      }
    };
  }
]).controller('TestDialogController', [
  '$scope', '$http', 'dialog', 'settings', function($scope, $http, dialog, settings) {
    var model;
    model = dialog.model;
    $scope.params = model.import_params;
    $scope.parameters = {};
    $scope.close = function() {
      return dialog.close();
    };
    return $scope.start = function(result) {
      var form_data, key;
      form_data = new FormData();
      for (key in $scope.parameters) {
        form_data.append(key, $scope.parameters[key]);
      }
      return $http({
        method: "POST",
        url: settings.apiUrl + ("model/" + model.name + "/test/test"),
        data: form_data,
        headers: {
          'Content-Type': void 0,
          'X-Requested-With': null
        },
        transformRequest: angular.identity
      }).success(function(data, status, headers, config) {
        $scope.success = true;
        $scope.msg = {};
        return dialog.close(result);
      }).error(function(data, status, headers, config) {
        return $scope.httpError = true;
      });
    };
  }
]).controller('Model_list', [
  '$scope', '$http', '$dialog', 'settings', function($scope, $http, $dialog, settings) {
    $scope.path = [
      {
        label: 'Home',
        url: '#/'
      }, {
        label: 'Models',
        url: '#/models'
      }
    ];
    $http({
      method: 'GET',
      url: settings.apiUrl + "model",
      headers: {
        'X-Requested-With': null
      }
    }).success(function(data, status, headers, config) {
      return $scope.models = data.models;
    }).error(function(data, status, headers, config) {
      return $scope.error = data;
    });
    return $scope.test = function(model) {
      var d;
      d = $dialog.dialog({
        modalFade: false
      });
      d.model = model;
      return d.open('partials/modal.html', 'TestDialogController');
    };
  }
]).controller('AddModelCtl', [
  '$scope', '$http', '$location', 'settings', function($scope, $http, $location, settings) {
    $scope.path = [
      {
        label: 'Home',
        url: '#/'
      }, {
        label: 'Train Model',
        url: '#/add_model'
      }
    ];
    $scope.upload = function() {
      var fd;
      fd = new FormData();
      fd.append("import_handler_local", $scope.import_handler_local);
      fd.append("features", $scope.features);
      return $http({
        method: "POST",
        url: settings.apiUrl + ("model/train/" + $scope.name),
        data: fd,
        headers: {
          'Content-Type': void 0,
          'X-Requested-With': null
        },
        transformRequest: angular.identity
      }).success(function(data, status, headers, config) {
        $scope.msg = data.name;
        return $location.path('/models');
      });
    };
    $scope.setImportHandlerLocalFile = function(element) {
      return $scope.$apply(function($scope) {
        $scope.msg = "";
        $scope.error = "";
        return $scope.import_handler_local = element.files[0];
      });
    };
    return $scope.setFeaturesFile = function(element) {
      return $scope.$apply(function($scope) {
        $scope.msg = "";
        $scope.error = "";
        return $scope.features = element.files[0];
      });
    };
  }
]).controller('UploadModelCtl', [
  '$scope', '$http', '$location', 'settings', function($scope, $http, $location, settings) {
    $scope.path = [
      {
        label: 'Home',
        url: '#/'
      }, {
        label: 'Upload Trained Model',
        url: '#/upload_model'
      }
    ];
    $scope.upload = function() {
      var fd;
      fd = new FormData();
      fd.append("file", $scope.file);
      fd.append("import_handler_local", $scope.import_handler_local);
      fd.append("features", $scope.features);
      return $http({
        method: "POST",
        url: settings.apiUrl + ("model/" + $scope.name),
        data: fd,
        headers: {
          'Content-Type': void 0,
          'X-Requested-With': null
        },
        transformRequest: angular.identity
      }).success(function(data, status, headers, config) {
        $scope.msg = data.name;
        return $location.path('/models');
      });
    };
    $scope.setFile = function(element) {
      return $scope.$apply(function($scope) {
        $scope.msg = "";
        $scope.error = "";
        return $scope.file = element.files[0];
      });
    };
    $scope.setImportHandlerLocalFile = function(element) {
      return $scope.$apply(function($scope) {
        $scope.msg = "";
        $scope.error = "";
        return $scope.import_handler_local = element.files[0];
      });
    };
    return $scope.setFeaturesFile = function(element) {
      return $scope.$apply(function($scope) {
        $scope.msg = "";
        $scope.error = "";
        return $scope.features = element.files[0];
      });
    };
  }
]).controller('ModelDetailsCtrl', [
  '$scope', '$http', '$routeParams', '$dialog', 'settings', function($scope, $http, $routeParams, $dialog, settings) {
    $scope.path = [
      {
        label: 'Home',
        url: '#/'
      }, {
        label: 'Models',
        url: '#/models'
      }, {
        label: 'Model Details',
        url: ''
      }
    ];
    $http({
      method: 'GET',
      url: settings.apiUrl + ("model/" + $routeParams.name),
      headers: {
        'X-Requested-With': null
      }
    }).success(function(data, status, headers, config) {
      $scope.model = data.model;
      return $scope.tests = data.tests;
    }).error(function(data, status, headers, config) {
      $scope.error = data;
      return $scope.httpError = true;
    });
    $scope.showWeightsTab = function() {
      return $scope.weights = {
        positive: $scope.model.positive_weights,
        negative: $scope.model.negative_weights
      };
    };
    return $scope.test = function(model) {
      var d;
      d = $dialog.dialog({
        modalFade: false
      });
      d.model = model;
      return d.open('partials/modal.html', 'TestDialogController');
    };
  }
]).controller('TestDetailsCtrl', [
  '$scope', '$http', '$routeParams', 'settings', function($scope, $http, $routeParams, settings) {
    $scope.path = [
      {
        label: 'Home',
        url: '#/'
      }, {
        label: 'Models',
        url: '#/models'
      }, {
        label: 'Model Details',
        url: ''
      }, {
        label: 'Test Details',
        url: ''
      }
    ];
    return $http({
      method: 'GET',
      url: settings.apiUrl + ("model/" + $routeParams.name + "/test/" + $routeParams.test_name),
      headers: {
        'X-Requested-With': null
      }
    }).success(function(data, status, headers, config) {
      $scope.test = data.test;
      $scope.metrics = data.metrics;
      return $scope.model = data.model;
    }).error(function(data, status, headers, config) {
      return $scope.error = data;
    });
  }
]).controller('TestExamplesCtrl', [
  '$scope', '$http', '$routeParams', 'settings', function($scope, $http, $routeParams, settings) {
    $scope.path = [
      {
        label: 'Home',
        url: '#/'
      }, {
        label: 'Models',
        url: '#/models'
      }, {
        label: 'Model Details',
        url: "#/models/" + $routeParams.name
      }, {
        label: 'Test Details',
        url: "#/models/" + $routeParams.name + "/tests/" + $routeParams.test_name
      }, {
        label: 'Test Examples',
        url: ''
      }
    ];
    $scope.currentPage = 1;
    return $scope.$watch('currentPage', function(currentPage, oldVal, scope) {
      return $http({
        method: 'GET',
        url: settings.apiUrl + ("model/" + $routeParams.name + "/test/" + $routeParams.test_name + "/data?page=" + currentPage),
        headers: {
          'X-Requested-With': null
        }
      }).success(function(data, status, headers, config) {
        $scope.data = data.data;
        $scope.test = data.test;
        return $scope.model = data.model;
      }).error(function(data, status, headers, config) {
        return $scope.error = data;
      });
    }, true);
  }
]).controller('ExampleDetailsCtrl', [
  '$scope', '$http', '$routeParams', 'settings', function($scope, $http, $routeParams, settings) {
    $scope.path = [
      {
        label: 'Home',
        url: '#/'
      }, {
        label: 'Models',
        url: '#/models'
      }, {
        label: 'Model Details',
        url: ''
      }, {
        label: 'Test Details',
        url: ''
      }, {
        label: 'Test Examples',
        url: ''
      }, {
        label: 'Example Details',
        url: ''
      }
    ];
    return $http({
      method: 'GET',
      url: settings.apiUrl + ("model/" + $routeParams.name + "/test/" + $routeParams.test_name + "/") + ("data/" + $routeParams.data_id),
      headers: {
        'X-Requested-With': null
      }
    }).success(function(data, status, headers, config) {
      $scope.data = data.data;
      $scope.test = data.test;
      return $scope.model = data.model;
    }).error(function(data, status, headers, config) {
      return $scope.error = data;
    });
  }
]);
'use strict';

/* Directives
*/

var createSVG, updateGraphPrecisionRecallCurve, updateGraphRocCurve, zip;

angular.module('app.directives', ['app.services']).directive('appVersion', [
  'version', function(version) {
    return function(scope, elm, attrs) {
      return elm.text(version);
    };
  }
]).directive('breadcrumb', function() {
  return {
    restrict: 'E',
    template: "<div><ul class='breadcrumb'><li ng-repeat='node in path'><a ng-href='{{node.url}}'>{{node.label}}</a><span class='divider'>/</span></li></ul><div ng-transclude></div></div>",
    replace: true,
    transclude: true,
    scope: {
      path: '='
    }
  };
}).directive('weightsTable', function() {
  return {
    restrict: 'E',
    template: '<table>\
                      <thead>\
                        <tr>\
                          <th>Paremeter</th>\
                          <th>Weight</th>\
                        </tr>\
                      </thead>\
                      <tbody>\
                        <tr ng-repeat="row in weights">\
                          <td>{{ row.name }}</td>\
                          <td>\
                            <div class="badge" ng-class="row.css_class">\
                              {{ row.weight }}</div>\
                          </td>\
                        </tr>\
                      </tbody>\
                    </table>',
    replace: true,
    transclude: true,
    scope: {
      weights: '='
    }
  };
}).directive("recursive", [
  '$compile', function($compile) {
    return {
      restrict: "EACM",
      priority: 100000,
      compile: function(tElement, tAttr) {
        var compiledContents, contents;
        contents = tElement.contents().remove();
        compiledContents = void 0;
        return function(scope, iElement, iAttr) {
          if (scope.row.full_name) {
            return;
          }
          if (!compiledContents) {
            compiledContents = $compile(contents);
          }
          return iElement.append(compiledContents(scope, function(clone) {
            return clone;
          }));
        };
      }
    };
  }
]).directive("tree", [
  function() {
    return {
      scope: {
        tree: '='
      },
      transclude: true,
      template: '<ul>\n          <li ng-repeat="(key, row) in tree" >\n            {{ key }}\n            <a ng-show="!row.value" ng-click="show=!show"\n              ng-init="show=false">\n<i ng-class="{false:\'icon-arrow-right\',true:\'icon-arrow-down\'}[show]"></i>\n            </a>\n            <span class="{{ row.css_class }}">{{ row.value }}</span>\n            <recursive ng-show="show">\n              <span tree="row"></span>\n            </recursive>\n          </li>\n        </ul>',
      compile: function() {
        return function() {};
      }
    };
  }
]).directive('scRocCurve', [
  function() {
    return {
      restrict: 'E',
      scope: {
        metrics: '='
      },
      link: function(scope, element, attrs) {
        createSVG(scope, element);
        return scope.$watch('metrics', updateGraphRocCurve, true);
      }
    };
  }
]).directive('scPrecisionRecallCurve', [
  function() {
    return {
      restrict: 'E',
      scope: {
        metrics: '='
      },
      link: function(scope, element, attrs) {
        createSVG(scope, element);
        return scope.$watch('metrics', updateGraphPrecisionRecallCurve, true);
      }
    };
  }
]);

createSVG = function(scope, element) {
  scope.margin = {
    top: 20,
    right: 20,
    bottom: 30,
    left: 20
  };
  scope.w = 400;
  scope.h = 300;
  if (!(scope.svg != null)) {
    return scope.svg = d3.select(element[0]).append("svg").attr("width", scope.w).attr("height", scope.h);
  }
};

updateGraphRocCurve = function(metrics, oldVal, scope) {
  var chart, data, rocCurve;
  data = [];
  if (!metrics) {
    return;
  }
  data = zip(metrics[0], metrics[1]);
  chart = nv.models.lineChart();
  chart.xAxis.axisLabel('False-positive rate').tickFormat(d3.format(',r'));
  chart.yAxis.axisLabel('True-positive rate').tickFormat(d3.format(',.2f'));
  rocCurve = function() {
    var i, line_data, roc_data, step, _i, _ref;
    roc_data = [];
    line_data = [];
    step = 1 / data.length;
    for (i = _i = 0, _ref = data.length; 0 <= _ref ? _i < _ref : _i > _ref; i = 0 <= _ref ? ++_i : --_i) {
      roc_data.push({
        x: data[i][0],
        y: data[i][1]
      });
      line_data.push({
        x: step * i,
        y: step * i
      });
    }
    return [
      {
        values: roc_data,
        key: "ROC Curve",
        color: "#000eff",
        "stroke-width": "10px"
      }, {
        values: line_data,
        key: "line",
        color: "red",
        "stroke-width": "1px",
        "stroke-dasharray": "10,10"
      }
    ];
  };
  scope.svg.datum(rocCurve()).transition().duration(500).call(chart);
  return nv.utils.windowResize(chart.update);
};

zip = function() {
  var arr, i, length, lengthArray, _i, _results;
  lengthArray = (function() {
    var _i, _len, _results;
    _results = [];
    for (_i = 0, _len = arguments.length; _i < _len; _i++) {
      arr = arguments[_i];
      _results.push(arr.length);
    }
    return _results;
  }).apply(this, arguments);
  length = Math.min.apply(Math, lengthArray);
  _results = [];
  for (i = _i = 0; 0 <= length ? _i < length : _i > length; i = 0 <= length ? ++_i : --_i) {
    _results.push((function() {
      var _j, _len, _results1;
      _results1 = [];
      for (_j = 0, _len = arguments.length; _j < _len; _j++) {
        arr = arguments[_j];
        _results1.push(arr[i]);
      }
      return _results1;
    }).apply(this, arguments));
  }
  return _results;
};

updateGraphPrecisionRecallCurve = function(metrics, oldVal, scope) {
  var chart, curve, data;
  data = [];
  if (!metrics) {
    return;
  }
  data = zip(metrics[0], metrics[1]);
  chart = nv.models.lineChart();
  chart.xAxis.axisLabel('Recall').tickFormat(d3.format(',r'));
  chart.yAxis.axisLabel('Precision').tickFormat(d3.format(',.2f'));
  curve = function() {
    var i, step, zipped_data, _i, _ref;
    zipped_data = [];
    step = 1 / data.length;
    for (i = _i = 0, _ref = data.length; 0 <= _ref ? _i < _ref : _i > _ref; i = 0 <= _ref ? ++_i : --_i) {
      zipped_data.push({
        x: data[i][0],
        y: data[i][1]
      });
    }
    return [
      {
        values: zipped_data,
        key: "Precision-Recall curve",
        color: "#000eff",
        "stroke-width": "10px"
      }
    ];
  };
  scope.svg.datum(curve()).transition().duration(500).call(chart);
  return nv.utils.windowResize(chart.update);
};
'use strict';

/* Filters
*/

var add_zero;

angular.module('app.filters', []).filter('interpolate', [
  'version', function(version) {
    return function(text) {
      return String(text).replace(/\%VERSION\%/mg, version);
    };
  }
]).filter('capfirst', [
  function() {
    return function(text) {
      var t;
      t = String(text);
      return t[0].toUpperCase() + t.slice(1);
    };
  }
]).filter('words', [
  function() {
    return function(text) {
      var t;
      t = String(text);
      return t.split(/\W+/);
    };
  }
]).filter('format_date', [
  function() {
    return function(text) {
      var d, dt, h, m, mm, y;
      dt = new Date(text);
      d = add_zero(dt.getDate());
      m = add_zero(dt.getMonth() + 1);
      y = dt.getFullYear();
      h = add_zero(dt.getHours());
      mm = add_zero(dt.getMinutes());
      return d + "-" + m + "-" + y + ' ' + h + ':' + mm;
    };
  }
]);

add_zero = function(val) {
  if (val < 10) {
    val = '0' + val;
  }
  return val;
};
'use strict';

/* Configuration specific to given environment
*/

var LOCAL_SETTINGS;

LOCAL_SETTINGS = {
  apiUrl: 'http://172.27.77.242/api/cloudml/b/v1/'
};

angular.module('app.local_config', []).constant('settings', LOCAL_SETTINGS);
'use strict';

/* Sevices
*/

angular.module('app.services', []).factory('version', function() {
  return "0.1";
});
