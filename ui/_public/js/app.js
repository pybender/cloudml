'use strict';

var App;

App = angular.module('app', ['ui', 'ngCookies', 'ngResource', 'app.config', 'app.controllers', 'app.directives', 'app.filters', 'app.services', 'ui.bootstrap', 'app.models.model', 'app.models.controllers', 'app.testresults.model', 'app.testresults.controllers', 'app.datas.model', 'app.datas.controllers']);

App.config([
  '$routeProvider', '$locationProvider', function($routeProvider, $locationProvider, config) {
    $routeProvider.when('/models', {
      controller: "ModelListCtrl",
      templateUrl: '/partials/model_list.html'
    }).when('/models/:name', {
      controller: 'ModelDetailsCtrl',
      templateUrl: '/partials/model_details.html'
    }).when('/models/:name/tests/:test_name', {
      controller: 'TestDetailsCtrl',
      templateUrl: '/partials/test_details.html'
    }).when('/models/:name/tests/:test_name/examples', {
      controller: 'TestExamplesCtrl',
      templateUrl: '/partials/test_examples.html'
    }).when('/models/:name/tests/:test_name/examples/:data_id', {
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

angular.module('app.controllers', ['app.config']).controller('AppCtrl', [
  '$scope', '$location', '$resource', '$rootScope', 'settings', function($scope, $location, $resource, $rootScope, settings) {
    $scope.$location = $location;
    $scope.pathElements = [];
    $scope.$watch('$location.path()', function(path) {
      return $scope.activeNavId = path || '/';
    });
    $scope.$on('$routeChangeSuccess', function(event, current) {
      var key, path, pathElement, pathElements, pathParamsLookup, result;
      pathElements = $location.path().split('/');
      result = [];
      path = '';
      pathElements.shift();
      pathParamsLookup = {};
      for (key in pathElements) {
        pathElement = pathElements[key];
        path += '/' + pathElement;
        result.push({
          name: pathElement,
          path: path
        });
      }
      return $scope.pathElements = result;
    });
    return $scope.getClass = function(id) {
      if ($scope.activeNavId.substring(0, id.length) === id) {
        return 'active';
      } else {
        return '';
      }
    };
  }
]).controller('ObjectListCtrl', [
  '$scope', function($scope) {
    var _this = this;
    $scope.pages = 0;
    $scope.page = 1;
    $scope.total = 0;
    $scope.per_page = 20;
    $scope.objects = [];
    $scope.loading = false;
    $scope.init = function(opts) {
      if (opts == null) {
        opts = {};
      }
      if (!_.isFunction(opts.objectLoader)) {
        throw new Error("Invalid object loader supplied to ObjectListCtrl");
      }
      $scope.objectLoader = opts.objectLoader;
      $scope.load();
      return $scope.$watch('page', function(page, oldVal, scope) {
        return $scope.load();
      }, true);
    };
    return $scope.load = function() {
      if ($scope.loading) {
        return false;
      }
      $scope.loading = true;
      return $scope.objectLoader({
        page: $scope.page
      }).then((function(opts) {
        $scope.loading = false;
        $scope.total = opts.total;
        $scope.page = opts.page;
        $scope.pages = opts.pages;
        $scope.per_page = opts.per_page;
        $scope.objects = opts.objects;
        return $scope.$broadcast('ObjectListCtrl:load:success', $scope.objects);
      }), (function(opts) {
        return $scope.$broadcast('ObjectListCtrl:load:error', opts);
      }));
    };
  }
]);
'use strict';

/* Tests examples specific Controllers
*/

angular.module('app.datas.controllers', ['app.config']).controller('TestExamplesCtrl', [
  '$scope', '$http', '$routeParams', 'settings', 'Data', function($scope, $http, $routeParams, settings, Data) {
    $scope.test_name = $routeParams.test_name;
    return $scope.loadDatas = function() {
      return function(pagination_opts) {
        return Data.$loadAll(_.extend({
          model_name: $routeParams.name,
          test_name: $routeParams.test_name
        }, pagination_opts));
      };
    };
  }
]).controller('ExampleDetailsCtrl', [
  '$scope', '$http', '$routeParams', 'settings', 'Data', function($scope, $http, $routeParams, settings, Data) {
    if (!$scope.data) {
      $scope.data = new Data({
        model_name: $routeParams.name,
        test_name: $routeParams.test_name,
        id: $routeParams.data_id
      });
    }
    return $scope.data.$load().then((function() {}), (function() {
      $scope.error = data;
      return $scope.httpError = true;
    }));
  }
]);
var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

angular.module('app.datas.model', ['app.config']).factory('Data', [
  '$http', '$q', 'settings', function($http, $q, settings) {
    var Data;
    Data = (function() {

      function Data(opts) {
        this.$load = __bind(this.$load, this);

        this.loadFromJSON = __bind(this.loadFromJSON, this);

        this.toJSON = __bind(this.toJSON, this);
        this.loadFromJSON(opts);
      }

      Data.prototype.id = null;

      Data.prototype.created_on = null;

      Data.prototype.model_name = null;

      Data.prototype.test_name = null;

      Data.prototype.data_input = null;

      Data.prototype.weighted_data_input = null;

      /* API methods
      */


      Data.prototype.isNew = function() {
        if (this.slug === null) {
          return true;
        } else {
          return false;
        }
      };

      Data.prototype.toJSON = function() {
        return {
          name: this.name
        };
      };

      Data.prototype.loadFromJSON = function(origData) {
        var data;
        data = _.extend({}, origData);
        return _.extend(this, data);
      };

      Data.prototype.$load = function() {
        var _this = this;
        if (this.name === null) {
          throw new Error("Can't load model without name");
        }
        return $http({
          method: 'GET',
          url: settings.apiUrl + ("model/" + this.model_name + "/test/" + this.test_name + "/data/" + this.id),
          headers: {
            'X-Requested-With': null
          }
        }).then((function(resp) {
          _this.loaded = true;
          _this.loadFromJSON(resp.data['data']);
          return resp;
        }), (function(resp) {
          return resp;
        }));
      };

      Data.$loadAll = function(opts) {
        var dfd, model_name, test_name,
          _this = this;
        dfd = $q.defer();
        model_name = opts.model_name;
        test_name = opts.test_name;
        $http({
          method: 'GET',
          url: "" + settings.apiUrl + "model/" + model_name + "/test/" + test_name + "/data",
          headers: settings.apiRequestDefaultHeaders,
          params: opts
        }).then((function(resp) {
          var extra, obj;
          extra = {
            loaded: true,
            model_name: model_name,
            test_name: test_name
          };
          return dfd.resolve({
            pages: resp.data['data'].pages,
            page: resp.data['data'].page,
            total: resp.data['data'].total,
            per_page: resp.data['data'].per_page,
            objects: (function() {
              var _i, _len, _ref, _results;
              _ref = resp.data['data'].items;
              _results = [];
              for (_i = 0, _len = _ref.length; _i < _len; _i++) {
                obj = _ref[_i];
                _results.push(new this(_.extend(obj, extra)));
              }
              return _results;
            }).call(_this),
            _resp: resp
          });
        }), (function() {
          return dfd.reject.apply(this, arguments);
        }));
        return dfd.promise;
      };

      return Data;

    })();
    return Data;
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
]).directive('showtab', function() {
  return {
    link: function(scope, element, attrs) {
      return element.click(function(e) {
        e.preventDefault();
        return $(element).tab('show');
      });
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
]).directive('loadindicator', function() {
  /*
      Usage::
  
        <loadindicator title="Loading jobs..." ng-show="!jobs" progress="'90%'">
        </loadindicator>
  
      Specify `progress` attribute if you want a progress bar. Value could be
      a string (enclosed in single quotes) or a function reference.
      It will be used as watch expression to dynamically update progress.
  
      If there's no `progress` attribute, then indicator will be simple ajaxy
      spinner.
  */
  return {
    restrict: 'E',
    replace: true,
    transclude: 'element',
    scope: true,
    template: '<div class="loading-indicator">\n</div>',
    link: function(scope, el, attrs) {
      var tmpl;
      if (attrs.progress) {
        tmpl = '<div class="progress progress-striped active">\n  <div class="bar" style="width: 100%;"></div>\n</div>';
        el.addClass('loading-indicator-progress').append($(tmpl));
        el.find('.bar').css({
          width: '0%'
        });
        return scope.$watch(attrs.progress, function(newVal, oldVal, scope) {
          return el.find('.bar').css({
            width: newVal
          });
        });
      } else {
        tmpl = '<img src="/img/ajax-loader.gif">';
        el.addClass('loading-indicator-spin');
        return el.append($(tmpl));
      }
    }
  };
}).directive('alert', function() {
  /*
      Use like this::
  
        <alert ng-show="savingError"
               alert-class="alert-error"
               msg="savingError" unsafe></alert>
  
      ``msg`` is an expression, and ``alert-class`` a string.
  
      ``unsafe`` is boolean, if present then contents retrieved from ``msg``
      are used to set the HTML content of the alert with all the markup.
  
      Important: NEVER pass user-generated content to ``msg`` with ``unsafe`` on.
  */
  return {
    restrict: 'E',
    replace: true,
    scope: true,
    template: '<div class="alert alert-block">\n  <button type="button"\n    class="close" data-dismiss="alert">&times;</button>\n  <div class="message"></div>\n</div>',
    link: function(scope, el, attrs) {
      var unsafe, _meth;
      unsafe = attrs.unsafe;
      _meth = unsafe === void 0 ? 'text' : 'html';
      el.find('.message')[_meth]('');
      attrs.$observe('msg', function(newVal, oldVal, scope) {
        if (newVal) {
          return el.find('.message')[_meth](newVal);
        }
      });
      return attrs.$observe('htmlclass', function(newVal, oldVal, scope) {
        var alert;
        alert = el;
        if (oldVal) {
          alert.removeClass(oldVal);
        }
        if (newVal) {
          return alert.addClass(newVal);
        }
      });
    }
  };
}).directive('scRocCurve', [
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
  data = zip(metrics[1], metrics[0]);
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
  apiUrl: 'http://127.0.0.1:5000/cloudml/b/v1/'
};

angular.module('app.local_config', []).constant('settings', LOCAL_SETTINGS);
'use strict';

/* Trained Model specific Controllers
*/

angular.module('app.models.controllers', ['app.config']).controller('ModelListCtrl', [
  '$scope', '$http', '$dialog', 'settings', 'Model', function($scope, $http, $dialog, settings, Model) {
    $scope.loadModels = function() {
      return function(pagination_opts) {
        return Model.$loadAll();
      };
    };
    $scope.test = function(model) {
      var d;
      d = $dialog.dialog({
        modalFade: false
      });
      d.model = model;
      return d.open('partials/modal.html', 'TestDialogController');
    };
    return $scope.train = function(model) {
      var d;
      d = $dialog.dialog({
        modalFade: false
      });
      d.model = model;
      return d.open('partials/model_train_popup.html', 'TrainModelCtrl');
    };
  }
]).controller('AddModelCtl', [
  '$scope', '$http', '$location', 'settings', 'Model', function($scope, $http, $location, settings, Model) {
    $scope.model = new Model();
    $scope["new"] = true;
    $scope.upload = function() {
      $scope.saving = true;
      $scope.savingProgress = '0%';
      $scope.savingError = null;
      _.defer(function() {
        $scope.savingProgress = '50%';
        return $scope.$apply();
      });
      return $scope.model.$save().then((function() {
        $scope.savingProgress = '100%';
        return _.delay((function() {
          $location.path('/models');
          return $scope.$apply();
        }), 300);
      }), (function(resp) {
        $scope.saving = false;
        return $scope.savingError = "Error while saving: server responded with " + ("" + resp.status + " (" + (resp.data.error || "no message") + "). ") + "Make sure you filled the form correctly. " + "Please contact support if the error will not go away.";
      }));
    };
    $scope.setImportHandlerFile = function(element) {
      return $scope.$apply(function($scope) {
        var reader;
        $scope.msg = "";
        $scope.error = "";
        $scope.import_handler = element.files[0];
        reader = new FileReader();
        reader.onload = function(e) {
          var str;
          str = e.target.result;
          return $scope.model.importhandler = str;
        };
        return reader.readAsText($scope.import_handler);
      });
    };
    $scope.setTrainImportHandlerFile = function(element) {
      return $scope.$apply(function($scope) {
        var reader;
        $scope.msg = "";
        $scope.error = "";
        $scope.train_import_handler = element.files[0];
        reader = new FileReader();
        reader.onload = function(e) {
          var str;
          str = e.target.result;
          return $scope.model.train_importhandler = str;
        };
        return reader.readAsText($scope.train_import_handler);
      });
    };
    return $scope.setFeaturesFile = function(element) {
      return $scope.$apply(function($scope) {
        var reader;
        $scope.msg = "";
        $scope.error = "";
        $scope.features = element.files[0];
        reader = new FileReader();
        reader.onload = function(e) {
          var str;
          str = e.target.result;
          return $scope.model.features = str;
        };
        return reader.readAsText($scope.features);
      });
    };
  }
]).controller('UploadModelCtl', [
  '$scope', '$http', '$location', 'settings', 'Model', function($scope, $http, $location, settings, Model) {
    $scope["new"] = true;
    $scope.model = new Model();
    $scope.upload = function() {
      $scope.saving = true;
      $scope.savingProgress = '0%';
      $scope.savingError = null;
      _.defer(function() {
        $scope.savingProgress = '50%';
        return $scope.$apply();
      });
      return $scope.model.$save().then((function() {
        $scope.savingProgress = '100%';
        return _.delay((function() {
          $location.path('/models');
          return $scope.$apply();
        }), 300);
      }), (function(resp) {
        $scope.saving = false;
        return $scope.savingError = "Error while saving: server responded with " + ("" + resp.status + " (" + (resp.data.error || "no message") + "). ") + "Make sure you filled the form correctly. " + "Please contact support if the error will not go away.";
      }));
    };
    $scope.setModelFile = function(element) {
      return $scope.$apply(function($scope) {
        $scope.msg = "";
        $scope.error = "";
        $scope.model_file = element.files[0];
        return $scope.model.trainer = element.files[0];
      });
    };
    return $scope.setImportHandlerFile = function(element) {
      return $scope.$apply(function($scope) {
        var reader;
        $scope.msg = "";
        $scope.error = "";
        $scope.import_handler = element.files[0];
        reader = new FileReader();
        reader.onload = function(e) {
          var str;
          str = e.target.result;
          return $scope.model.importhandler = str;
        };
        return reader.readAsText($scope.import_handler);
      });
    };
  }
]).controller('ModelDetailsCtrl', [
  '$scope', '$http', '$location', '$routeParams', '$dialog', 'settings', 'Model', 'TestResult', function($scope, $http, $location, $routeParams, $dialog, settings, Model, Test) {
    var DEFAULT_ACTION,
      _this = this;
    if (!$scope.model) {
      if (!$routeParams.name) {
        throw new Error("Can't initialize model detail controller      without model name");
      }
      $scope.model = new Model({
        name: $routeParams.name
      });
    }
    $scope.model.$load().then((function() {
      return $scope.latest_test = new Test($scope.model.latest_test);
    }), (function() {
      $scope.error = data;
      return $scope.httpError = true;
    }));
    DEFAULT_ACTION = 'model:details';
    $scope.action = ($routeParams.action || DEFAULT_ACTION).split(':');
    $scope.$watch('action', function(action) {
      var actionString;
      actionString = action.join(':');
      return $location.search(actionString === DEFAULT_ACTION ? "" : "action=" + actionString);
    });
    $scope.toggleAction = function(action) {
      return $scope.action = action;
    };
    $scope.loadTests = function() {
      return function(pagination_opts) {
        return Test.$loadTests($scope.model.name);
      };
    };
    $scope.saveImportHandlerChanges = function() {
      if (!$scope.importHandlerChanged) {
        return false;
      }
      return $scope.model.$save({
        only: ['importhandler']
      }).then((function() {
        return $scope.importHandlerChanged = false;
      }), (function() {
        throw new Error("Unable to save import handler");
      }));
    };
    $scope.$watch('model.importhandler', function(newVal, oldVal) {
      if ((newVal != null) && (oldVal != null) && newVal !== "" && oldVal !== "") {
        return $scope.importHandlerChanged = true;
      }
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
]).controller('TrainModelCtrl', [
  '$scope', '$http', 'dialog', 'settings', function($scope, $http, dialog, settings) {
    $scope.model = dialog.model;
    $scope.params = $scope.model.import_params;
    $scope.parameters = {};
    $scope.close = function() {
      return dialog.close();
    };
    return $scope.start = function(result) {
      return $scope.model.$train($scope.parameters).then((function() {
        return $scope.close();
      }), (function() {
        throw new Error("Unable to start model training");
      }));
    };
  }
]);
var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

angular.module('app.models.model', ['app.config']).factory('Model', [
  '$http', '$q', 'settings', function($http, $q, settings) {
    var Model, trimTrailingWhitespace;
    trimTrailingWhitespace = function(val) {
      return val.replace(/^\s+|\s+$/g, '');
    };
    /*
        Trained Model
    */

    Model = (function() {

      function Model(opts) {
        this.$train = __bind(this.$train, this);

        this.$save = __bind(this.$save, this);

        this.prepareSaveJSON = __bind(this.prepareSaveJSON, this);

        this.$load = __bind(this.$load, this);

        this.loadFromJSON = __bind(this.loadFromJSON, this);

        this.toJSON = __bind(this.toJSON, this);
        this.loadFromJSON(opts);
      }

      Model.prototype.id = null;

      Model.prototype.created_on = null;

      Model.prototype.status = null;

      Model.prototype.name = null;

      Model.prototype.trainer = null;

      Model.prototype.importParams = null;

      Model.prototype.negative_weights = null;

      Model.prototype.negative_weights_tree = null;

      Model.prototype.positive_weights = null;

      Model.prototype.positive_weights_tree = null;

      Model.prototype.latest_test = null;

      Model.prototype.importhandler = null;

      Model.prototype.train_importhandler = null;

      Model.prototype.features = null;

      /* API methods
      */


      Model.prototype.isNew = function() {
        if (this.id === null) {
          return true;
        } else {
          return false;
        }
      };

      Model.prototype.toJSON = function() {
        return {
          importhandler: this.importhandler,
          trainer: this.trainer,
          features: this.features
        };
      };

      Model.prototype.loadFromJSON = function(origData) {
        var data;
        data = _.extend({}, origData);
        return _.extend(this, data);
      };

      Model.prototype.$load = function() {
        var _this = this;
        if (this.name === null) {
          throw new Error("Can't load model without name");
        }
        return $http({
          method: 'GET',
          url: settings.apiUrl + ("model/" + this.name),
          headers: {
            'X-Requested-With': null
          }
        }).then((function(resp) {
          _this.loaded = true;
          _this.loadFromJSON(resp.data['model']);
          return resp;
        }), (function(resp) {
          return resp;
        }));
      };

      Model.prototype.prepareSaveJSON = function(json) {
        var reqData;
        reqData = json || this.toJSON();
        return reqData;
      };

      Model.prototype.$save = function(opts) {
        var fd,
          _this = this;
        if (opts == null) {
          opts = {};
        }
        fd = new FormData();
        fd.append("trainer", this.trainer);
        fd.append("importhandler", this.importhandler);
        fd.append("train_importhandler", this.train_importhandler);
        fd.append("features", this.features);
        return $http({
          method: this.isNew() ? "POST" : "PUT",
          headers: {
            'Content-Type': void 0,
            'X-Requested-With': null
          },
          url: "" + settings.apiUrl + "model/" + (this.name || ""),
          data: fd,
          transformRequest: angular.identity
        }).then(function(resp) {
          return _this.loadFromJSON(resp.data['model']);
        });
      };

      Model.$loadAll = function(opts) {
        var dfd,
          _this = this;
        dfd = $q.defer();
        $http({
          method: 'GET',
          url: "" + settings.apiUrl + "model/",
          headers: settings.apiRequestDefaultHeaders
        }).then((function(resp) {
          var obj;
          return dfd.resolve({
            total: resp.data.found,
            objects: (function() {
              var _i, _len, _ref, _results;
              _ref = resp.data.models;
              _results = [];
              for (_i = 0, _len = _ref.length; _i < _len; _i++) {
                obj = _ref[_i];
                _results.push(new this(_.extend(obj, {
                  loaded: true
                })));
              }
              return _results;
            }).call(_this),
            _resp: resp
          });
        }), (function() {
          return dfd.reject.apply(this, arguments);
        }));
        return dfd.promise;
      };

      Model.prototype.$train = function(opts) {
        var fd, key, val,
          _this = this;
        if (opts == null) {
          opts = {};
        }
        fd = new FormData();
        for (key in opts) {
          val = opts[key];
          fd.append(key, val);
        }
        return $http({
          method: "PUT",
          headers: {
            'Content-Type': void 0,
            'X-Requested-With': null
          },
          url: "" + settings.apiUrl + "model/" + this.name + "/train",
          data: fd,
          transformRequest: angular.identity
        }).then(function(resp) {
          return _this.loadFromJSON(resp.data['model']);
        });
      };

      return Model;

    })();
    return Model;
  }
]);
'use strict';

/* Sevices
*/

angular.module('app.services', []).factory('version', function() {
  return "0.1";
});
'use strict';

/* Tests specific Controllers
*/

angular.module('app.testresults.controllers', ['app.config']).controller('TestDialogController', [
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
]).controller('TestDetailsCtrl', [
  '$scope', '$http', '$routeParams', 'settings', 'TestResult', function($scope, $http, $routeParams, settings, Test) {
    if (!$scope.test) {
      if (!$routeParams.name) {
        throw new Error("Can't initialize test detail controller      without test name");
      }
      $scope.test = new Test({
        model_name: $routeParams.name,
        name: $routeParams.test_name
      });
    }
    return $scope.test.$load().then((function() {}), (function() {
      $scope.error = data;
      return $scope.httpError = true;
    }));
  }
]);
var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
  __indexOf = [].indexOf || function(item) { for (var i = 0, l = this.length; i < l; i++) { if (i in this && this[i] === item) return i; } return -1; };

angular.module('app.testresults.model', ['app.config']).factory('TestResult', [
  '$http', '$q', 'settings', 'Model', function($http, $q, settings, Model) {
    var TestResult, trimTrailingWhitespace;
    trimTrailingWhitespace = function(val) {
      return val.replace(/^\s+|\s+$/g, '');
    };
    /*
        Trained Model
    */

    TestResult = (function() {

      function TestResult(opts) {
        this.$save = __bind(this.$save, this);

        this.$load = __bind(this.$load, this);

        this.loadFromJSON = __bind(this.loadFromJSON, this);

        this.toJSON = __bind(this.toJSON, this);
        this.loadFromJSON(opts);
      }

      TestResult.prototype.id = null;

      TestResult.prototype.accuracy = null;

      TestResult.prototype.created_on = null;

      TestResult.prototype.data_count = null;

      TestResult.prototype.name = null;

      TestResult.prototype.parameters = null;

      TestResult.prototype.model = null;

      TestResult.prototype.model_name = null;

      /* API methods
      */


      TestResult.prototype.isNew = function() {
        if (this.slug === null) {
          return true;
        } else {
          return false;
        }
      };

      TestResult.prototype.toJSON = function() {
        return {
          name: this.name
        };
      };

      TestResult.prototype.loadFromJSON = function(origData) {
        var data;
        data = _.extend({}, origData);
        return _.extend(this, data);
      };

      TestResult.prototype.$load = function() {
        var _this = this;
        if (this.name === null) {
          throw new Error("Can't load model without name");
        }
        return $http({
          method: 'GET',
          url: settings.apiUrl + ("model/" + this.model_name + "/test/" + this.name),
          headers: {
            'X-Requested-With': null
          }
        }).then((function(resp) {
          _this.loaded = true;
          _this.loadFromJSON(resp.data['test']);
          _this.model = new Model(resp.data['model']);
          return resp;
        }), (function(resp) {
          return resp;
        }));
      };

      TestResult.prototype.$save = function(opts) {
        var fields, key, saveData, _i, _len, _ref,
          _this = this;
        if (opts == null) {
          opts = {};
        }
        saveData = this.toJSON();
        fields = opts.only || [];
        if (fields.length > 0) {
          _ref = _.keys(saveData);
          for (_i = 0, _len = _ref.length; _i < _len; _i++) {
            key = _ref[_i];
            if (__indexOf.call(fields, key) < 0) {
              delete saveData[key];
            }
          }
        }
        saveData = this.prepareSaveJSON(saveData);
        return $http({
          method: this.isNew() ? 'POST' : 'PUT',
          headers: settings.apiRequestDefaultHeaders,
          url: "" + settings.apiUrl + "/jobs/" + (this.id || ""),
          params: {
            access_token: user.access_token
          },
          data: $.param(saveData)
        }).then(function(resp) {
          return _this.loadFromJSON(resp.data);
        });
      };

      TestResult.$loadTests = function(modelName, opts) {
        var dfd,
          _this = this;
        dfd = $q.defer();
        if (!modelName) {
          throw new Error("Model is required to load tests");
        }
        $http({
          method: 'GET',
          url: "" + settings.apiUrl + "model/" + modelName + "/tests",
          headers: settings.apiRequestDefaultHeaders,
          params: _.extend({}, opts)
        }).then((function(resp) {
          var obj;
          return dfd.resolve({
            total: resp.data.found,
            objects: (function() {
              var _i, _len, _ref, _results;
              _ref = resp.data.tests;
              _results = [];
              for (_i = 0, _len = _ref.length; _i < _len; _i++) {
                obj = _ref[_i];
                _results.push(new this(obj));
              }
              return _results;
            }).call(_this),
            _resp: resp
          });
        }), (function() {
          return dfd.reject.apply(this, arguments);
        }));
        return dfd.promise;
      };

      return TestResult;

    })();
    return TestResult;
  }
]);
