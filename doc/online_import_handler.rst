======================
Online import handlers
======================

Online import handlers using for getting data in cloudml-predict.

For getting data using following datasources:

   - Http
   - Input
  
Http datasource use to request data from another services. Input datasource for use data which get from POST of request.

Http
~~~~

For example::

    <plan>
	  <inputs>
	    <param name="opening_id" type="string"/>
	  </inputs>
	  <datasources>
	    <http name="odr" url="http://odr.prod.sv4.odesk.com/odr/opening/f/"/>
	  </datasources>
	  <import>
	    <entity datasource="odr" name="opening">
	      <query><![CDATA[#{opening_id}.json]]></query>
	      <field jsonpath="$.op_title" name="opening.title" type="string"/>
	      <field jsonpath="$.op_job" name="opening.description" type="string"/>
	    </entity>
	  </import>
	  <predict>
	    <result>
	      <label/>
	      <probability/>
	    </result>
	  </predict>
	</plan>


Input
~~~~~

For example::

	<plan>
	  <inputs>
	    <param name="rate" type="float"/>
	    <param name="title" type="string"/>
	  </inputs>
	  <datasources/>
	  <import>
	    <entity datasource="input" name="Test_input">
	      <query><![CDATA[any]]></query>
	      <field column="title" name="dev_title" type="string"/>
	      <field column="rate" name="rate" type="float"/>
	    </entity>
	  </import>
	  <predict>
	    <result>
	      <label/>
	      <probability/>
	    </result>
	  </predict>
	</plan>
