"use strict";var App;App=angular.module("app",["ui","ngCookies","ngResource","app.config","app.controllers","app.directives","app.filters","app.services","ui.bootstrap","app.models.model","app.models.controllers","app.testresults.model","app.testresults.controllers","app.datas.model","app.datas.controllers","app.reports.model","app.reports.controllers","app.importhandlers.model","app.importhandlers.controllers"]),App.config(["$routeProvider","$locationProvider",function(e,t,n){return e.when("/models",{controller:"ModelListCtrl",templateUrl:"/partials/model_list.html"}).when("/models/:name",{controller:"ModelDetailsCtrl",templateUrl:"/partials/model_details.html"}).when("/models/:name/tests/:test_name",{controller:"TestDetailsCtrl",templateUrl:"/partials/test_details.html"}).when("/models/:name/tests/:test_name/examples",{controller:"TestExamplesCtrl",templateUrl:"/partials/test_examples.html",reloadOnSearch:!1}).when("/models/:name/tests/:test_name/grouped_examples",{controller:"GroupedExamplesCtrl",templateUrl:"/partials/grouped_examples.html"}).when("/models/:name/tests/:test_name/examples/:data_id",{controller:"ExampleDetailsCtrl",templateUrl:"/partials/example_details.html"}).when("/upload_model",{templateUrl:"/partials/upload_model.html",controller:"UploadModelCtl"}).when("/add_model",{templateUrl:"/partials/add_model.html",controller:"AddModelCtl"}).when("/compare_models",{templateUrl:"/partials/compare_models_form.html",controller:"CompareModelsFormCtl"}).when("/import_handlers",{controller:"ImportHandlerListCtrl",templateUrl:"/partials/import_handler/list.html"}).when("/import_handlers/add",{controller:"AddImportHandlerCtl",templateUrl:"/partials/import_handler/add.html"}).when("/import_handlers/:name",{controller:"ImportHandlerDetailsCtrl",templateUrl:"/partials/import_handler/details.html"}).otherwise({redirectTo:"/models"}),t.html5Mode(!1)}]),App.run(["$rootScope",function(e){return e.Math=window.Math,e.includeLibraries=!0,e.include=function(e){var t,n;n=this;for(t in e)n[t]=getLibrary(t);return n}}]),"use strict";var SETTINGS;SETTINGS={apiUrl:"http://172.27.77.141/api/cloudml/",apiRequestDefaultHeaders:{"Content-Type":"application/x-www-form-urlencoded","X-Requested-With":null}},angular.module("app.config",[]).config(["$provide",function(e){var t,n;try{angular.module("app.local_config")}catch(r){console.warn("Couldn't find local settings: "+r.message),angular.module("app.local_config",[]).constant("settings",{})}return n=angular.injector(["app.local_config"]),t=n.get("settings"),e.constant("settings",$.extend(SETTINGS,t))}]),"use strict",angular.module("app.controllers",["app.config"]).controller("AppCtrl",["$scope","$location","$resource","$rootScope","settings",function(e,t,n,r,i){return e.$location=t,e.pathElements=[],e.$watch("$location.path()",function(t){return e.activeNavId=t||"/"}),e.$on("$routeChangeSuccess",function(n,r){var i,s,o,u,a,f;u=t.path().split("/"),f=[],s="",u.shift(),a={};for(i in u)o=u[i],s+="/"+o,f.push({name:o,path:s});return e.pathElements=f}),e.getClass=function(t){return e.activeNavId.substring(0,t.length)===t?"active":""}}]).controller("ObjectListCtrl",["$scope",function(e){var t=this;return e.pages=0,e.page=1,e.total=0,e.per_page=20,e.objects=[],e.loading=!1,e.init=function(t){t==null&&(t={});if(!_.isFunction(t.objectLoader))throw new Error("Invalid object loader supplied to ObjectListCtrl");return e.objectLoader=t.objectLoader,e.$watch("page",function(t,n,r){return e.load()},!0),e.$watch("filter_opts ",function(t,n,r){return e.load()},!0)},e.load=function(){return e.loading?!1:(e.loading=!0,e.objectLoader({page:e.page,filter_opts:e.filter_opts}).then(function(t){return e.loading=!1,e.total=t.total,e.page=t.page||1,e.pages=t.pages,e.per_page=t.per_page,e.objects=t.objects,e.$broadcast("ObjectListCtrl:load:success",e.objects)},function(t){return e.$broadcast("ObjectListCtrl:load:error",t)}))}}]),"use strict",angular.module("app.datas.controllers",["app.config"]).controller("TestExamplesCtrl",["$scope","$http","$routeParams","settings","$location","Data","Model","TestResult",function(e,t,n,r,i,s,o,u){return e.test_name=n.test_name,e.filter_opts=i.search(),e.model=new o({name:n.name}),e.test=new u({model_name:n.name,name:n.test_name}),e.model.$load({show:"name,labels"}).then(function(){return e.labels=e.model.labels,e.$watch("filter_opts",function(e,t,n){return i.search(e)},!0)},function(){return e.err=data}),e.loadDatas=function(){return function(e){var t;return t=e.filter_opts,delete e.filter_opts,s.$loadAll(_.extend({model_name:n.name,test_name:n.test_name,show:"name,label,pred_label,title"},e,t))}}}]).controller("GroupedExamplesCtrl",["$scope","$http","$routeParams","settings","Data",function(e,t,n,r,i){return e.test_name=n.test_name,e.model_name=n.name,e.form={field:"data_input.hire_outcome",count:100},e.update=function(){return i.$loadAllGroupped({model_name:n.name,test_name:n.test_name,field:e.form.field,count:e.form.count}).then(function(t){return e.field_name=t.field_name,e.mavp=t.mavp,e.objects=t.objects},function(t){return e.err="Error while loading: server responded with "+(""+t.status+" ")+("("+(t.data.response.error.message||"no message")+").")})},e.update()}]).controller("ExampleDetailsCtrl",["$scope","$http","$routeParams","settings","Data",function(e,t,n,r,i){return e.data||(e.data=new i({model_name:n.name,test_name:n.test_name,id:n.data_id})),e.data.$load({show:"id,weighted_data_input,target_variable,pred_label,label"}).then(function(){},function(){return e.error=data,e.httpError=!0})}]);var __bind=function(e,t){return function(){return e.apply(t,arguments)}};angular.module("app.datas.model",["app.config"]).factory("Data",["$http","$q","settings",function(e,t,n){var r;return r=function(){function r(e){this.loadFromJSON=__bind(this.loadFromJSON,this),this.toJSON=__bind(this.toJSON,this),this.loadFromJSON(e)}return r.prototype.id=null,r.prototype.created_on=null,r.prototype.model_name=null,r.prototype.test_name=null,r.prototype.data_input=null,r.prototype.weighted_data_input=null,r.prototype._id=null,r.prototype.isNew=function(){return this.slug===null?!0:!1},r.prototype.toJSON=function(){return{name:this.name}},r.prototype.loadFromJSON=function(e){var t;return t=_.extend({},e),_.extend(this,t)},r.prototype.$load=function(t){var r=this;if(this.name===null)throw new Error("Can't load model without name");return e({method:"GET",url:n.apiUrl+("model/"+this.model_name+"/test/"+this.test_name+"/data/"+this.id),headers:{"X-Requested-With":null},params:_.extend({},t)}).then(function(e){return r.loaded=!0,r.loadFromJSON(e.data.data),e},function(e){return e})},r.$loadAll=function(r){var i,s,o,u=this;return i=t.defer(),s=r.model_name,o=r.test_name,e({method:"GET",url:""+n.apiUrl+"model/"+s+"/test/"+o+"/data",headers:n.apiRequestDefaultHeaders,params:r}).then(function(e){var t,n;return t={loaded:!0,model_name:s,test_name:o},i.resolve({pages:e.data.pages,page:e.data.page,total:e.data.total,per_page:e.data.per_page,objects:function(){var r,i,s,o;s=e.data.datas,o=[];for(r=0,i=s.length;r<i;r++)n=s[r],o.push(new this(_.extend(n,t)));return o}.call(u),_resp:e})},function(){return i.reject.apply(this,arguments)}),i.promise},r.$loadAllGroupped=function(r){var i,s=this;return i=t.defer(),e({method:"GET",url:""+n.apiUrl+"model/"+r.model_name+"/test/"+r.test_name+"/action/groupped/data?field="+r.field+"&count="+r.count,headers:n.apiRequestDefaultHeaders,params:r}).then(function(e){return i.resolve({field_name:e.data.field_name,mavp:e.data.mavp,objects:e.data.datas.items})},function(){return i.reject.apply(this,arguments)}),i.promise},r}(),r}]),"use strict";var createSVG,getPlotData,updateCurves,zip;angular.module("app.directives",["app.services"]).directive("appVersion",["version",function(e){return function(t,n,r){return n.text(e)}}]).directive("showtab",function(){return{link:function(e,t,n){return t.click(function(e){return e.preventDefault(),$(t).tab("show")})}}}).directive("weightsTable",function(){return{restrict:"E",template:'<table>                      <thead>                        <tr>                          <th>Paremeter</th>                          <th>Weight</th>                        </tr>                      </thead>                      <tbody>                        <tr ng-repeat="row in weights">                          <td>{{ row.name }}</td>                          <td>                            <div class="badge" ng-class="row.css_class">                              {{ row.weight }}</div>                          </td>                        </tr>                      </tbody>                    </table>',replace:!0,transclude:!0,scope:{weights:"="}}}).directive("weightedDataParameters",function(){return{restrict:"E",template:'<span>\n<span ng-show="!val.weights" title="weight={{ val.weight }}"\nclass="badge {{ val.css_class }}">{{ val.value }}</span>\n\n<div ng-show="val.weights">\n  <span  ng-show="val.type == \'List\'"\n  ng-init="lword=word.toLowerCase()"\n  ng-repeat="word in val.value|words">\n    <span ng-show="val.weights[lword].weight"\n    title="weight={{ val.weights[lword].weight }}"\n    class="badge {{ val.weights[lword].css_class }}">{{ word }}</span>\n    <span ng-show="!val.weights[lword].weight">{{ word }}</span></span>\n\n  <span ng-show="val.type == \'Dictionary\'"\n  ng-repeat="(key, dval) in val.weights">\n    <span title="weight={{ dval.weight }}"\n    class="badge {{ dval.css_class }}">\n      {{ key }}={{ dval.value }}</span></span>\n</div>\n</span>',replace:!0,transclude:!0,scope:{val:"="}}}).directive("confusionMatrix",function(){return{restrict:"E",templateUrl:"partials/directives/confusion_matrix.html",scope:{matrix:"=",url:"="},replace:!0,transclude:!0}}).directive("recursive",["$compile",function(e){return{restrict:"EACM",priority:1e5,compile:function(t,n){var r,i;return i=t.contents().remove(),r=void 0,function(t,n,s){if(t.row.full_name)return;return r||(r=e(i)),n.append(r(t,function(e){return e}))}}}}]).directive("tree",[function(){return{scope:{tree:"="},transclude:!0,template:'<ul>\n          <li ng-repeat="(key, row) in tree" >\n            {{ key }}\n            <a ng-show="!row.value" ng-click="show=!show"\n              ng-init="show=false">\n<i ng-class="{false:\'icon-arrow-right\',true:\'icon-arrow-down\'}[show]"></i>\n            </a>\n            <span class="{{ row.css_class }}">{{ row.value }}</span>\n            <recursive ng-show="show">\n              <span tree="row"></span>\n            </recursive>\n          </li>\n        </ul>',compile:function(){return function(){}}}}]).directive("loadindicator",function(){return{restrict:"E",replace:!0,transclude:"element",scope:!0,template:'<div class="loading-indicator">\n</div>',link:function(e,t,n){var r;return n.progress?(r='<div class="progress progress-striped active">\n  <div class="bar" style="width: 100%;"></div>\n</div>',t.addClass("loading-indicator-progress").append($(r)),t.find(".bar").css({width:"0%"}),e.$watch(n.progress,function(e,n,r){return t.find(".bar").css({width:e})})):(r='<img src="/img/ajax-loader.gif">',t.addClass("loading-indicator-spin"),t.append($(r)))}}}).directive("alertMessage",function(){return{restrict:"E",replace:!0,scope:!0,template:'<div class="alert alert-block">\n  <button type="button"\n    class="close" data-dismiss="alert">&times;</button>\n  <div class="message"></div>\n</div>',link:function(e,t,n){var r,i;return r=n.unsafe,i=r===void 0?"text":"html",t.find(".message")[i](""),n.$observe("msg",function(e,n,r){if(e)return t.find(".message")[i](e)}),n.$observe("htmlclass",function(e,n,r){var i;i=t,n&&i.removeClass(n);if(e)return i.addClass(e)})}}}).directive("scCurves",[function(){return{restrict:"E",scope:{curvesDict:"=",xLabel:"@xlabel",yLabel:"@ylabel",showLine:"@showLine",width:"@width",height:"@height"},link:function(e,t,n){return createSVG(e,t,n.width,n.height),e.$watch("curvesDict",updateCurves)}}}]),createSVG=function(e,t,n,r){n==null&&(n=400),r==null&&(r=300),e.margin={top:20,right:20,bottom:30,left:210};if(e.svg==null)return e.svg=d3.select(t[0]).append("svg").attr("width",n).attr("height",r)},updateCurves=function(e,t,n){var r;if(!e)return;return r=nv.models.lineChart(),r.xAxis.orient("bottom").axisLabel(n.xLabel).tickFormat(d3.format(",r")),r.yAxis.orient("left").axisLabel(n.yLabel).tickFormat(d3.format(",.f")),n.svg.datum(getPlotData(e,n.showLine)).transition().duration(500).call(r),nv.utils.windowResize(r.update)},zip=function(){var e,t,n,r,i,s;r=function(){var t,n,r;r=[];for(t=0,n=arguments.length;t<n;t++)e=arguments[t],r.push(e.length);return r}.apply(this,arguments),n=Math.min.apply(Math,r),s=[];for(t=i=0;0<=n?i<n:i>n;t=0<=n?++i:--i)s.push(function(){var n,r,i;i=[];for(n=0,r=arguments.length;n<r;n++)e=arguments[n],i.push(e[t]);return i}.apply(this,arguments));return s},getPlotData=function(e,t){var n,r,i,s,o,u,a,f;u=[];for(o in e)s=e[o],s!=null&&s[0]!=null&&s[1]!=null&&(n=zip(s[0],s[1]),a=1/n.length,f=function(){var e,t,i;i=[];for(r=e=0,t=n.length;0<=t?e<t:e>t;r=0<=t?++e:--e)i.push({x:n[r][0],y:n[r][1]});return i}(),u.push({values:f,key:o}));return t&&n!=null&&(i=function(){var e,t,i;i=[];for(r=e=0,t=n.length;0<=t?e<t:e>t;r=0<=t?++e:--e)i.push({x:a*r,y:a*r});return i}(),u.push({values:i,key:"line"})),u},"use strict";var add_zero;angular.module("app.filters",[]).filter("interpolate",["version",function(e){return function(t){return String(t).replace(/\%VERSION\%/mg,e)}}]).filter("capfirst",[function(){return function(e){var t;return t=String(e),t[0].toUpperCase()+t.slice(1)}}]).filter("words",[function(){return function(e){var t;return t=String(e),t.split(/\W+/)}}]).filter("range",[function(){return function(e,t){var n,r,i;t=parseInt(t);for(n=r=0,i=t-1;0<=i?r<=i:r>=i;n=0<=i?++r:--r)e.push(n);return e}}]).filter("format_date",[function(){return function(e){var t,n,r,i,s,o;return n=new Date(e),t=add_zero(n.getDate()),i=add_zero(n.getMonth()+1),o=n.getFullYear(),r=add_zero(n.getHours()),s=add_zero(n.getMinutes()),t+"-"+i+"-"+o+" "+r+":"+s}}]),add_zero=function(e){return e<10&&(e="0"+e),e},"use strict",angular.module("app.importhandlers.controllers",["app.config"]).controller("ImportHandlerListCtrl",["$scope","$http","$dialog","settings","ImportHandler",function(e,t,n,r,i){return i.$loadAll({show:"name,type,created_on,updated_on"}).then(function(t){return e.objects=t.objects},function(t){return e.err="Error while saving: server responded with "+(""+resp.status+" ")+("("+(resp.data.response.error.message||"no message")+"). ")+"Make sure you filled the form correctly. "+"Please contact support if the error will not go away."})}]).controller("ImportHandlerDetailsCtrl",["$scope","$http","$location","$routeParams","$dialog","settings","ImportHandler",function(e,t,n,r,i,s,o){var u;return e.model||(r.name||(u="Can't initialize without import handler name"),e.handler=new o({name:r.name})),e.handler.$load({show:"name,type,created_on,updated_on,data"}).then(function(){var e;e=!0;if(typeof callback!="undefined"&&callback!==null)return callback()},function(){return e.err=data})}]).controller("AddImportHandlerCtl",["$scope","$http","$location","settings","ImportHandler",function(e,t,n,r,i){return e.handler=new i,e.types=[{name:"Db"},{name:"Request"}],e.err="",e["new"]=!0,e.add=function(){return e.saving=!0,e.savingProgress="0%",e.savingError=null,_.defer(function(){return e.savingProgress="50%",e.$apply()}),e.handler.$save().then(function(){return e.savingProgress="100%",_.delay(function(){return n.path(e.handler.objectUrl()),e.$apply()},300)},function(t){return e.saving=!1,e.err="Error while saving: server responded with "+(""+t.status+" ")+("("+(t.data.response.error.message||"no message")+"). ")+"Make sure you filled the form correctly. "+"Please contact support if the error will not go away."})},e.setDataFile=function(t){return e.$apply(function(e){var n;return e.msg="",e.error="",e.data=t.files[0],n=new FileReader,n.onload=function(t){var n;return n=t.target.result,e.handler.data=n},n.readAsText(e.data)})}}]);var __bind=function(e,t){return function(){return e.apply(t,arguments)}};angular.module("app.importhandlers.model",["app.config"]).factory("ImportHandler",["$http","$q","settings",function(e,t,n){var r;return r=function(){function r(e){this.$save=__bind(this.$save,this),this.loadFromJSON=__bind(this.loadFromJSON,this),this.objectUrl=__bind(this.objectUrl,this),this.loadFromJSON(e)}return r.prototype.id=null,r.prototype.created_on=null,r.prototype.updated_on=null,r.prototype.name=null,r.prototype.type=null,r.prototype.data=null,r.prototype.objectUrl=function(){return"/import_handlers/"+this.name},r.prototype.isNew=function(){return this.id===null?!0:!1},r.prototype.loadFromJSON=function(e){var t,n;t=_.extend({},e),_.extend(this,t);if(e!=null)return this.data=angular.toJson(e.data,n=!0)},r.prototype.$load=function(t){var r=this;if(this.name===null)throw new Error("Can't load import handler model without name");return e({method:"GET",url:""+n.apiUrl+"import/handler/"+this.name,headers:{"X-Requested-With":null},params:_.extend({},t)}).then(function(e){return r.loaded=!0,r.loadFromJSON(e.data.import_handler),e},function(e){return e})},r.prototype.$save=function(t){var r,i=this;return t==null&&(t={}),r=new FormData,r.append("name",this.name),r.append("type",this.type.name),r.append("data",this.data),e({method:this.isNew()?"POST":"PUT",headers:{"Content-Type":void 0,"X-Requested-With":null},url:""+n.apiUrl+"import/handler/"+(this.name||""),data:r,transformRequest:angular.identity}).then(function(e){return i.loadFromJSON(e.data.import_handler)})},r.$loadAll=function(r){var i,s=this;return i=t.defer(),e({method:"GET",url:""+n.apiUrl+"import/handler/",headers:n.apiRequestDefaultHeaders,params:_.extend({},r)}).then(function(e){var t;return i.resolve({total:e.data.found,objects:function(){var n,r,i,s;i=e.data.import_handlers,s=[];for(n=0,r=i.length;n<r;n++)t=i[n],s.push(new this(_.extend(t,{loaded:!0})));return s}.call(s),_resp:e})},function(){return i.reject.apply(this,arguments)}),i.promise},r}(),r}]),"use strict";var LOCAL_SETTINGS;LOCAL_SETTINGS={apiUrl:"http://172.27.77.141/api/cloudml/"},angular.module("app.local_config",[]).constant("settings",LOCAL_SETTINGS),"use strict",angular.module("app.models.controllers",["app.config"]).controller("ModelListCtrl",["$scope","$http","$dialog","settings","Model",function(e,t,n,r,i){return i.$loadAll({show:"name,status,created_on,import_params,error"}).then(function(t){return e.objects=t.objects},function(t){return e.err="Error while saving: server responded with "+(""+t.status+" ")+("("+(t.data.response.error.message||"no message")+"). ")+"Make sure you filled the form correctly. "+"Please contact support if the error will not go away."})}]).controller("AddModelCtl",["$scope","$http","$location","settings","Model",function(e,t,n,r,i){return e.model=new i,e.err="",e["new"]=!0,e.upload=function(){return e.saving=!0,e.savingProgress="0%",e.savingError=null,_.defer(function(){return e.savingProgress="50%",e.$apply()}),e.model.$save().then(function(){return e.savingProgress="100%",_.delay(function(){return n.path(e.model.objectUrl()),e.$apply()},300)},function(t){return e.saving=!1,e.err="Error while saving: server responded with "+(""+t.status+" ")+("("+(t.data.response.error.message||"no message")+"). ")+"Make sure you filled the form correctly. "+"Please contact support if the error will not go away."})},e.setImportHandlerFile=function(t){return e.$apply(function(e){var n;return e.msg="",e.error="",e.import_handler=t.files[0],n=new FileReader,n.onload=function(t){var n;return n=t.target.result,e.model.importhandler=n,e.model.train_importhandler=n},n.readAsText(e.import_handler)})},e.setFeaturesFile=function(t){return e.$apply(function(e){var n;return e.msg="",e.error="",e.features=t.files[0],n=new FileReader,n.onload=function(t){var n;return n=t.target.result,e.model.features=n},n.readAsText(e.features)})}}]).controller("UploadModelCtl",["$scope","$http","$location","settings","Model",function(e,t,n,r,i){return e["new"]=!0,e.model=new i,e.upload=function(){return e.saving=!0,e.savingProgress="0%",e.savingError=null,_.defer(function(){return e.savingProgress="50%",e.$apply()}),e.model.$save().then(function(){return e.savingProgress="100%",_.delay(function(){return n.path(e.model.objectUrl()),e.$apply()},300)},function(t){return e.saving=!1,e.err="Error while saving: server responded with "+(""+t.status+" ")+("("+(t.data.response.error.message||"no message")+"). ")+"Make sure you filled the form correctly. "+"Please contact support if the error will not go away."})},e.setModelFile=function(t){return e.$apply(function(e){return e.msg="",e.error="",e.model_file=t.files[0],e.model.trainer=t.files[0]})},e.setImportHandlerFile=function(t){return e.$apply(function(e){var n;return e.msg="",e.error="",e.import_handler=t.files[0],n=new FileReader,n.onload=function(t){var n;return n=t.target.result,e.model.importhandler=n},n.readAsText(e.import_handler)})}}]).controller("ModelDetailsCtrl",["$scope","$http","$location","$routeParams","$dialog","settings","Model","TestResult",function(e,t,n,r,i,s,o,u){var a,f,l=this;return a="model:details",e.ppage=1,e.npage=1,e.positive=[],e.negative=[],e.action=(r.action||a).split(":"),e.$watch("action",function(t){var r;r=t.join(":"),n.search(r===a?"":"action="+r);switch(t[0]){case"features":return e.go("features,status");case"weights":return e.goWeights(!0,!0);case"test":return e.goTests();case"import_handlers":return t[1]==="train"?e.go("train_importhandler,status,id"):e.go("importhandler,status,id");default:return e.go("status,created_on,target_variable,error")}}),e.model||(r.name||(f="Can't initialize without model name"),e.model=new o({name:r.name})),e.toggleAction=function(t){return e.action=t},e.go=function(t,n){return e.model.$load({show:t}).then(function(){var e;e=!0;if(n!=null)return n()},function(){return e.err=data})},e.goWeights=function(t,n){return e.model.$loadWeights({show:"status,name",ppage:e.ppage,npage:e.npage}).then(function(r){t&&e.positive.push.apply(e.positive,e.model.positive_weights);if(n)return e.negative.push.apply(e.negative,e.model.negative_weights)},function(){return e.err=data})},e.morePositiveWeights=function(){return e.ppage+=1,e.goWeights(!0,!1)},e.moreNegativeWeights=function(){return e.npage+=1,e.goWeights(!1,!0)},e.goTests=function(){return u.$loadTests(e.model.name,{show:"name,created_on,status,parameters,accuracy,examples_count"}).then(function(t){return e.tests=t.objects},function(t){return e.err="Error while loading tests: server responded with "+(""+t.status+" ")+("("+(t.data.response.error.message||"no message")+"). ")})},e.saveTrainHandler=function(){return e.model.$save({only:["train_importhandler"]}).then(function(){return e.msg="Import Handler for training model saved"},function(){throw new Error("Unable to save import handler")})},e.saveTestHandler=function(){return e.model.$save({only:["importhandler"]}).then(function(){return e.msg="Import Handler for tests saved"},function(){throw new Error("Unable to save import handler")})}}]).controller("TrainModelCtrl",["$scope","$http","dialog","settings",function(e,t,n,r){return e.model=n.model,e.model.$load({show:"import_params"}).then(function(){return e.params=e.model.import_params},function(){return e.err=data}),e.parameters={},e.close=function(){return n.close()},e.start=function(t){return e.model.$train(e.parameters).then(function(){return e.close()},function(){throw new Error("Unable to start model training")})}}]).controller("DeleteModelCtrl",["$scope","$http","dialog","settings","$location",function(e,t,n,r,i){return e.model=n.model,e.close=function(){return n.close()},e["delete"]=function(t){return e.model.$delete().then(function(){return e.close(),i.path("#/models")},function(t){return t.data?e.err="Error while deleting model:server responded with "+(""+t.status+" ")+("("+(t.data.response.error.message||"no message")+"). "):e.err="Error while deleting model"})}}]).controller("ModelActionsCtrl",["$scope","$dialog",function(e,t){var n=this;return e.init=function(t){t==null&&(t={});if(!t.model)throw new Error("Please specify model");return e.model=t.model},e.test_model=function(e){var n;return n=t.dialog({modalFade:!1}),n.model=e,n.open("partials/modal.html","TestDialogController")},e.train_model=function(e){var n;return n=t.dialog({modalFade:!1}),n.model=e,n.open("partials/model_train_popup.html","TrainModelCtrl")},e.delete_model=function(e){var n;return n=t.dialog({modalFade:!1}),n.model=e,n.open("partials/models/delete_model_popup.html","DeleteModelCtrl")}}]);var __bind=function(e,t){return function(){return e.apply(t,arguments)}};angular.module("app.models.model",["app.config"]).factory("Model",["$http","$q","settings",function(e,t,n){var r,i;return i=function(e){return e.replace(/^\s+|\s+$/g,"")},r=function(){function r(e){this.$train=__bind(this.$train,this),this.$delete=__bind(this.$delete,this),this.$save=__bind(this.$save,this),this.prepareSaveJSON=__bind(this.prepareSaveJSON,this),this.loadFromJSON=__bind(this.loadFromJSON,this),this.toJSON=__bind(this.toJSON,this),this.objectUrl=__bind(this.objectUrl,this),this.loadFromJSON(e)}return r.prototype._id=null,r.prototype.created_on=null,r.prototype.status=null,r.prototype.name=null,r.prototype.trainer=null,r.prototype.importParams=null,r.prototype.negative_weights=null,r.prototype.negative_weights_tree=null,r.prototype.positive_weights=null,r.prototype.positive_weights_tree=null,r.prototype.latest_test=null,r.prototype.importhandler=null,r.prototype.train_importhandler=null,r.prototype.features=null,r.prototype.objectUrl=function(){return"/models/"+this.name},r.prototype.isNew=function(){return this._id===null?!0:!1},r.prototype.toJSON=function(){return{importhandler:this.importhandler,trainer:this.trainer,features:this.features}},r.prototype.loadFromJSON=function(e){var t,n;t=_.extend({},e),_.extend(this,t);if(e!=null)return this.created_on=String(e.created_on),this.features=angular.toJson(e.features,n=!0),this.importhandler=angular.toJson(e.importhandler,n=!0),this.train_importhandler=angular.toJson(e.train_importhandler,n=!0)},r.prototype.$load=function(t){var r=this;if(this.name===null)throw new Error("Can't load model without name");return e({method:"GET",url:n.apiUrl+("model/"+this.name),headers:{"X-Requested-With":null},params:_.extend({},t)}).then(function(e){return r.loaded=!0,r.loadFromJSON(e.data.model),e},function(e){return e})},r.prototype.prepareSaveJSON=function(e){var t;return t=e||this.toJSON(),t},r.prototype.$save=function(t){var r,i=this;return t==null&&(t={}),r=new FormData,r.append("trainer",this.trainer),r.append("importhandler",this.importhandler),r.append("train_importhandler",this.train_importhandler),r.append("features",this.features),e({method:this.isNew()?"POST":"PUT",headers:{"Content-Type":void 0,"X-Requested-With":null},url:""+n.apiUrl+"model/"+(this.name||""),data:r,transformRequest:angular.identity}).then(function(e){return i.loadFromJSON(e.data.model)})},r.prototype.$delete=function(t){return t==null&&(t={}),e({method:"DELETE",headers:{"Content-Type":void 0,"X-Requested-With":null},url:""+n.apiUrl+"model/"+this.name,transformRequest:angular.identity})},r.$loadAll=function(r){var i,s=this;return i=t.defer(),e({method:"GET",url:""+n.apiUrl+"model/",headers:n.apiRequestDefaultHeaders,params:_.extend({},r)}).then(function(e){var t;return i.resolve({total:e.data.found,objects:function(){var n,r,i,s;i=e.data.models,s=[];for(n=0,r=i.length;n<r;n++)t=i[n],s.push(new this(_.extend(t,{loaded:!0})));return s}.call(s),_resp:e})},function(){return i.reject.apply(this,arguments)}),i.promise},r.prototype.$loadWeights=function(t){var r=this;if(this.name===null)throw new Error("Can't load model without name");return e({method:"GET",url:n.apiUrl+("model/"+this.name+"/weights"),headers:{"X-Requested-With":null},params:_.extend({},t)}).then(function(e){return r.loaded=!0,r.loadFromJSON(e.data.model),e},function(e){return e})},r.prototype.$train=function(t){var r,i,s,o=this;t==null&&(t={}),r=new FormData;for(i in t)s=t[i],r.append(i,s);return e({method:"PUT",headers:{"Content-Type":void 0,"X-Requested-With":null},url:""+n.apiUrl+"model/"+this.name+"/train",data:r,transformRequest:angular.identity}).then(function(e){return o.loadFromJSON(e.data.model)})},r}(),r}]),"use strict",angular.module("app.reports.controllers",["app.config"]).controller("CompareModelsFormCtl",["$scope","$http","$location","$routeParams","settings","Model","TestResult","Data","CompareReport",function(e,t,n,r,i,s,o,u,a){var f,l,c=this;return f="form:",e.section="metrics",e.action=(r.action||f).split(":"),e.$watch("action",function(t){var r,i,s,o,u,l,c,h;i=t[1].split(",");if(e.report==null&&i.length!==0){o={};for(s=c=0,h=i.length;0<=h?c<h:c>h;s=0<=h?++c:--c)l=i[s],u=Math.floor(s/2+1),s%2===1?o["test_name"+u]=l:o["model_name"+u]=l;e.report=new a(o)}return t[0]==="report"?e.report.generated||e.generate():t[0]==="form"&&e.initForm(),r=t.join(":"),n.search(r===f?"":"action="+r)}),l=function(t,n,r){if(t!=null)return e.loadTestsList(t)},e.$watch("model1",l,!0),e.$watch("model2",l,!0),e.is_form=function(){return e.action[0]==="form"},e.loadModelsList=function(){return s.$loadAll({comparable:1}).then(function(t){var n,r,i,s,o;e.models=t.objects;if(e.is_form()){s=e.models,o=[];for(r=0,i=s.length;r<i;r++)n=s[r],n.name===e.report.model_name1&&(e.model1=n),n.name===e.report.model_name2?o.push(e.model2=n):o.push(void 0);return o}},function(e){var t;return t=e.$error})},e.loadTestsList=function(t){return o.$loadTests(t.name,{status:"Completed"}).then(function(n){var r,i,s,o,u;t.tests=n.objects;if(e.is_form()){o=t.tests,u=[];for(i=0,s=o.length;i<s;i++)r=o[i],r.name===e.report.test_name1&&t.name===e.report.model_name1&&(e.test1=r),r.name===e.report.test_name2&&t.name===e.report.model_name2?u.push(e.test2=r):u.push(void 0);return u}},function(e){var t;return t=e.$error})},e.backToForm=function(){return e.toogleAction("form")},e.generateReport=function(){var t;return t={test_name1:e.test1.name,model_name1:e.model1.name,test_name2:e.test2.name,model_name2:e.model2.name},e.report=new a(t),e.toogleAction("report","metrics")},e.toogleAction=function(t){var n;return n=e.report,e.action=[t,""+n.model_name1+","+n.test_name1+","+(""+n.model_name2+","+n.test_name2)]},e.toogleReportSection=function(t){return e.section=t},e.initForm=function(){return e.loadModelsList()},e.generate=function(){return e.generating=!0,e.generatingProgress="0%",e.generatingError=null,_.defer(function(){return e.generatingProgress="70%",e.$apply()}),e.report.$getReportData().then(function(){return e.generatingProgress="100%",e.generating=!1,e.generated=!0},function(t){return e.generating=!1,e.err="Error while generating compare report:"+("server responded with "+t.status+" ")+("("+(t.data.response.error.message||"no message")+"). ")+"Make sure you filled the form correctly. "+"Please contact support if the error will not go away."})}}]);var __bind=function(e,t){return function(){return e.apply(t,arguments)}};angular.module("app.reports.model",["app.config"]).factory("CompareReport",["$http","$q","settings","Model","TestResult","Data",function($http,$q,settings,Model,Test,Data){var CompareReport;return CompareReport=function(){function CompareReport(e){this.$getReportData=__bind(this.$getReportData,this),this.loadFromJSON=__bind(this.loadFromJSON,this),this.objectUrl=__bind(this.objectUrl,this),this.loadFromJSON(e)}return CompareReport.prototype.generated=!1,CompareReport.prototype.objectUrl=function(){if(this.model!=null)return"/models/"+this.model.name+"/tests/"+this.name},CompareReport.prototype.loadFromJSON=function(e){var t;return t=_.extend({},e),_.extend(this,t)},CompareReport.prototype.$getReportData=function(){var params,_this=this;return params={test1:this.test_name1,test2:this.test_name2,model1:this.model_name1,model2:this.model_name2},$http({method:"GET",url:settings.apiUrl+"reports/compare",headers:{"X-Requested-With":null},params:params}).then(function(resp){var auc,example,examples,examples_data,key,num,test,value,_i,_len,_ref;_this.generated=!0,_this.tests=[],_this.rocCurves={},_this.precisionRecallCurves={},_ref=resp.data;for(key in _ref){value=_ref[key],key.indexOf("test")===0&&(test=new Test(resp.data[key]),_this.tests.push(test),eval("_this."+key+"=test"),auc=test.metrics.roc_auc.toFixed(4),_this.rocCurves[test.fullName()+" (AUC = "+auc+")"]=test.metrics.roc_curve,_this.precisionRecallCurves[test.fullName()]=test.metrics.precision_recall_curve.reverse());if(key.indexOf("examples")===0){num=key.replace("examples",""),examples=[],examples_data=resp.data[key];for(_i=0,_len=examples_data.length;_i<_len;_i++)example=examples_data[_i],examples.push(new Data(example));eval("_this.test"+num+".examples=examples")}}return resp},function(e){return e})},CompareReport}(),CompareReport}]),"use strict",angular.module("app.services",[]).factory("version",function(){return"0.1"}),"use strict",angular.module("app.testresults.controllers",["app.config"]).controller("TestDialogController",["$scope","$http","dialog","settings","$location","TestResult",function(e,t,n,r,i,s){return e.model=n.model,e.model.$load({show:"import_params"}).then(function(){return e.params=e.model.import_params},function(){return e.err=data}),e.parameters={},e.close=function(){return n.close()},e.start=function(i){var s,o,u;s=new FormData,u=e.model;for(o in e.parameters)s.append(o,e.parameters[o]);return t({method:"POST",url:r.apiUrl+("model/"+u.name+"/test/test"),data:s,headers:{"Content-Type":void 0,"X-Requested-With":null},transformRequest:angular.identity}).success(function(t,r,s,o){return e.success=!0,t.test.model_name=u.name,n.close(i)}).error(function(t,n,r,i){return e.httpError=!0})}}]).controller("DeleteTestCtrl",["$scope","$http","dialog","settings","$location",function(e,t,n,r,i){return e.test=n.test,e.model=n.test.model,e.close=function(){return n.close()},e["delete"]=function(t){return e.test.$delete().then(function(){return e.close(),i.search("action=test:list&any="+Math.random())},function(t){return t.data?e.err="Error while deleting test:server responded with "+(""+t.status+" ")+("("+(t.data.response.error.message||"no message")+"). "):e.err="Error while deleting test"})}}]).controller("TestDetailsCtrl",["$scope","$http","$routeParams","settings","TestResult","$location",function(e,t,n,r,i,s){var o;if(!e.test){if(!n.name)throw new Error("Can't initialize test detail controller      without test name");e.test=new i({model_name:n.name,name:n.test_name}),e.test_num=n.test_name}return o="test:details",e.action=(n.action||o).split(":"),e.$watch("action",function(t){var n;n=t.join(":"),s.search(n===o?"":"action="+n);switch(t[0]){case"curves":return e.goMetrics();case"matrix":return e.go("status,metrics.confusion_matrix");default:return e.go("name,status,classes_set,created_on,accuracy,parameters,error,examples_count")}}),e.go=function(t,n){return e.test.$load({show:t}).then(function(){var e;e=!0;if(n!=null)return n()},function(){return e.err="Error"})},e.goMetrics=function(t,n){var r=this;return e.go("status,metrics.roc_curve,metrics.precision_recall_curve,metrics.roc_auc",function(){var t;return e.rocCurve={"ROC curve":e.test.metrics.roc_curve},t=e.test.metrics.precision_recall_curve,e.prCurve={"Precision-Recall curve":[t[1],t[0]]}})}}]).controller("TestActionsCtrl",["$scope","$dialog",function(e,t){var n=this;return e.init=function(t){var n,r;r=t.test,n=t.model;if(!r||!n)throw new Error("Please specify test and model");return t.test.model=n,e.test=r},e.delete_test=function(n){var r;return r=t.dialog({modalFade:!1}),r.test=e.test,r.open("partials/testresults/delete_popup.html","DeleteTestCtrl")}}]);var __bind=function(e,t){return function(){return e.apply(t,arguments)}},__indexOf=[].indexOf||function(e){for(var t=0,n=this.length;t<n;t++)if(t in this&&this[t]===e)return t;return-1};angular.module("app.testresults.model",["app.config"]).factory("TestResult",["$http","$q","settings","Model",function(e,t,n,r){var i;return i=function(){function i(e){this.$delete=__bind(this.$delete,this),this.$save=__bind(this.$save,this),this.loadFromJSON=__bind(this.loadFromJSON,this),this.toJSON=__bind(this.toJSON,this),this.fullName=__bind(this.fullName,this),this.examplesUrl=__bind(this.examplesUrl,this),this.objectUrl=__bind(this.objectUrl,this),this.loadFromJSON(e)}return i.prototype._id=null,i.prototype.accuracy=null,i.prototype.created_on=null,i.prototype.data_count=null,i.prototype.name=null,i.prototype.parameters=null,i.prototype.model=null,i.prototype.model_name=null,i.prototype.loaded=!1,i.prototype.isNew=function(){return this._id===null?!0:!1},i.prototype.objectUrl=function(){return"/models/"+(this.model_name||this.model.name)+"/tests/"+this.name},i.prototype.examplesUrl=function(){var e;return e=this.model_name||this.model.name,"/models/"+e+"/tests/"+this.name+"/examples"},i.prototype.fullName=function(){return this.model!=null||this.model_name?(this.model_name||this.model.name)+" / "+this.name:this.name},i.prototype.toJSON=function(){return{name:this.name}},i.prototype.loadFromJSON=function(e){var t;t=_.extend({},e),_.extend(this,t);if(__indexOf.call(e,"model")>=0)return this.model=new r(e.model),this.model_name=e.model.name},i.prototype.$load=function(t){var r=this;if(this.name===null)throw new Error("Can't load model without name");return e({method:"GET",url:n.apiUrl+("model/"+this.model_name+"/test/"+this.name),headers:{"X-Requested-With":null},params:_.extend({},t)}).then(function(e){return r.loaded=!0,r.loadFromJSON(e.data.test),e},function(e){return e})},i.prototype.$save=function(t){var r,i,s,o,u,a,f=this;t==null&&(t={}),s=this.toJSON(),r=t.only||[];if(r.length>0){a=_.keys(s);for(o=0,u=a.length;o<u;o++)i=a[o],__indexOf.call(r,i)<0&&delete s[i]}return s=this.prepareSaveJSON(s),e({method:this.isNew()?"POST":"PUT",headers:n.apiRequestDefaultHeaders,url:""+n.apiUrl+"/jobs/"+(this.id||""),params:{access_token:user.access_token},data:$.param(s)}).then(function(e){return f.loadFromJSON(e.data)})},i.prototype.$delete=function(t){return t==null&&(t={}),e({method:"DELETE",headers:n.apiRequestDefaultHeaders,url:""+n.apiUrl+"model/"+this.model.name+"/test/"+this.name,transformRequest:angular.identity})},i.$loadTests=function(r,s){var o,u=this;o=t.defer();if(!r)throw new Error("Model is required to load tests");return e({method:"GET",url:""+n.apiUrl+"model/"+r+"/tests",headers:n.apiRequestDefaultHeaders,params:_.extend({},s)}).then(function(e){var t;return o.resolve({objects:function(){var n,r,s,o;s=e.data.tests,o=[];for(n=0,r=s.length;n<r;n++)t=s[n],o.push(new i(t));return o}(),_resp:e})},function(){return o.reject.apply(this,arguments)}),o.promise},i}(),i}])