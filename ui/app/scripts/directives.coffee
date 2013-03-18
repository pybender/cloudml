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


.directive('loadindicator',

  () ->

    ###
    Usage::

      <loadindicator title="Loading jobs..." ng-show="!jobs" progress="'90%'">
      </loadindicator>

    Specify `progress` attribute if you want a progress bar. Value could be
    a string (enclosed in single quotes) or a function reference.
    It will be used as watch expression to dynamically update progress.

    If there's no `progress` attribute, then indicator will be simple ajaxy
    spinner.
    ###

    return {
      restrict: 'E'
      replace: true
      transclude: 'element'
      scope: true
      template: '''
        <div class="loading-indicator">
        </div>
        '''

      link: (scope, el, attrs) ->

        # Show progress bar if progress attribute is specified
        if attrs.progress
          tmpl = '''
            <div class="progress progress-striped active">
              <div class="bar" style="width: 100%;"></div>
            </div>
            '''
          el.addClass('loading-indicator-progress').append $(tmpl)

          el.find('.bar').css width: '0%'
          # Progress attribute value is expected to be a valid watchExpression
          # because it is going to be watched for changes
          scope.$watch attrs.progress, (newVal, oldVal, scope) ->
            el.find('.bar').css width: newVal

        # Spinner otherwise
        else
          tmpl = '''
            <img src="/img/ajax-loader.gif">
            '''
          el.addClass 'loading-indicator-spin'
          el.append $(tmpl)
    }
)


.directive('alert',

  () ->
    ###
    Use like this::

      <alert ng-show="savingError"
             alert-class="alert-error"
             msg="savingError" unsafe></alert>

    ``msg`` is an expression, and ``alert-class`` a string.

    ``unsafe`` is boolean, if present then contents retrieved from ``msg``
    are used to set the HTML content of the alert with all the markup.

    Important: NEVER pass user-generated content to ``msg`` with ``unsafe`` on.
    ###
    return {
      restrict: 'E'
      replace: true
      scope: true

      template: '''
        <div class="alert alert-block">
          <button type="button"
            class="close" data-dismiss="alert">&times;</button>
          <div class="message"></div>
        </div>
        '''

      link: (scope, el, attrs) ->

        unsafe = attrs.unsafe
        _meth = if unsafe is undefined then 'text' else 'html'

        el.find('.message')[_meth] ''
        attrs.$observe 'msg', (newVal, oldVal, scope) ->
          if newVal
            el.find('.message')[_meth] newVal

        attrs.$observe 'htmlclass', (newVal, oldVal, scope) ->
          alert = el

          if oldVal
            alert.removeClass oldVal

          if newVal
            alert.addClass newVal
    }
)

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

  data = zip(metrics[1], metrics[0])
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
