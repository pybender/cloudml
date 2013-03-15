'use strict'

### Directives ###

# register the module with Angular
angular.module('app.directives', [
  # require the 'app.service' module
  'app.services'
])

.directive('appVersion', [
  'version'

(version) ->

  (scope, elm, attrs) ->
    elm.text(version)
])

.directive('breadcrumb', () ->
  return {
    restrict: 'E',
    template: "<div><ul class='breadcrumb'><li ng-repeat='node in path'>
<a ng-href='{{node.url}}'>{{node.label}}</a>
<span class='divider'>/</span></li></ul>
<div ng-transclude></div></div>",
    replace: true,
    transclude : true,
    scope: { path: '=' }
  }
)

.directive('showtab', () ->
  return {
    link: (scope, element, attrs) ->
      element.click((e) ->
        e.preventDefault()
        $(element).tab('show')
      )
  }
)


.directive('weightsTable', () ->
  return {
    restrict: 'E',
    template: '<table>
                      <thead>
                        <tr>
                          <th>Paremeter</th>
                          <th>Weight</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr ng-repeat="row in weights">
                          <td>{{ row.name }}</td>
                          <td>
                            <div class="badge" ng-class="row.css_class">
                              {{ row.weight }}</div>
                          </td>
                        </tr>
                      </tbody>
                    </table>',
    replace: true,
    transclude : true,
    scope: { weights: '=' }
  }
)

.directive("recursive", [
  '$compile'

($compile) ->
  return {
    restrict: "EACM"
    priority: 100000
    compile: (tElement, tAttr) ->
      contents = tElement.contents().remove()
      compiledContents = undefined
      return (scope, iElement, iAttr) ->
        if scope.row.full_name
          return
        if not compiledContents
          compiledContents = $compile(contents)
        iElement.append(
          compiledContents(scope, (clone) -> return clone))
  }
])

.directive("tree", [ ->
  return {
    scope: {tree: '='}
    # replace: true
    #restrict: 'E'
    transclude : true
    template: '''<ul>
                <li ng-repeat="(key, row) in tree" >
                  {{ key }}
                  <a ng-show="!row.value" ng-click="show=!show"
                    ng-init="show=false">
      <i ng-class="{false:'icon-arrow-right',true:'icon-arrow-down'}[show]"></i>
                  </a>
                  <span class="{{ row.css_class }}">{{ row.value }}</span>
                  <recursive ng-show="show">
                    <span tree="row"></span>
                  </recursive>
                </li>
              </ul>'''
    compile: () ->
      return () ->
  }
])

.directive('scRocCurve', [ ->
  return {
    restrict: 'E',
    scope: { metrics: '=' },
    link: (scope, element, attrs) ->
      createSVG(scope, element)
      scope.$watch('metrics', updateGraphRocCurve, true)
  }
])

.directive('scPrecisionRecallCurve', [ ->
  return {
    restrict: 'E',
    scope: { metrics: '=' },
    link: (scope, element, attrs) ->
      createSVG(scope, element)
      scope.$watch('metrics', updateGraphPrecisionRecallCurve, true)
  }
])

createSVG = (scope, element) ->
  scope.margin = {top: 20, right: 20, bottom: 30, left: 20}
  scope.w = 400
  scope.h = 300
 
  if not scope.svg?
    scope.svg = d3.select(element[0])
    .append("svg")
    .attr("width", scope.w)
    .attr("height", scope.h)

updateGraphRocCurve = (metrics, oldVal, scope) ->
  data = []
  if !metrics
    return

  data = zip(metrics[0], metrics[1])
  chart = nv.models.lineChart()

  chart.xAxis
  .axisLabel('False-positive rate')
  .tickFormat(d3.format(',r'))

  chart.yAxis
  .axisLabel('True-positive rate')
  .tickFormat(d3.format(',.2f'))

  rocCurve = ->
    roc_data = []
    line_data = []
    step = 1 / data.length
    for i in [0...data.length]
      roc_data.push({x: data[i][0], y: data[i][1] })
      line_data.push({x: step*i, y: step*i })
    return [
      {
        values: roc_data,
        key: "ROC Curve",
        color: "#000eff",
        "stroke-width": "10px"
      },
      {
        values: line_data,
        key: "line",
        color: "red",
        "stroke-width": "1px",
        "stroke-dasharray": "10,10"
      }
    ]

  scope.svg.datum(rocCurve())
  .transition().duration(500)
  .call(chart)

  nv.utils.windowResize(chart.update)

zip = () ->
  lengthArray = (arr.length for arr in arguments)
  length = Math.min(lengthArray...)
  for i in [0...length]
    arr[i] for arr in arguments

updateGraphPrecisionRecallCurve = (metrics, oldVal, scope) ->
  data = []
  if !metrics
    return

  data = zip(metrics[0], metrics[1])
  chart = nv.models.lineChart()

  chart.xAxis
  .axisLabel('Recall')
  .tickFormat(d3.format(',r'))

  chart.yAxis
  .axisLabel('Precision')
  .tickFormat(d3.format(',.2f'))

  curve = ->
    zipped_data = []
    step = 1 / data.length
    for i in [0...data.length]
      zipped_data.push({x: data[i][0], y: data[i][1] })
    return [
      {
        values: zipped_data,
        key: "Precision-Recall curve",
        color: "#000eff",
        "stroke-width": "10px"
      }
    ]

  scope.svg.datum(curve())
  .transition().duration(500)
  .call(chart)

  nv.utils.windowResize(chart.update)