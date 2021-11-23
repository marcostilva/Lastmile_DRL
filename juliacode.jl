#########################################################################
# Julia Code to be used integrated with Python for 
# Deep Reinforcement Learning Algorithm for Last mile delivery with crowd shipping
# 

using JuMP, CPLEX, Combinatorics, Dates, Statistics, MathProgBase, DataStructures  

##################################################################
# Create Tree Structure for Branch-cut-and-price algorithm

#type TreeNode...Used only for DRO algorithm
struct TreeNode
  parent::Int
  children::Vector{Int}
  addedsolutions::Vector{Int}
  addedcenarios::Vector{Int}
  varxfixedsolutionzero::Vector{Vector{Int}}
  varxfixedsolutionone::Vector{Vector{Int}}
  vardfixedsolutionzero::Vector{Int}
  vardfixedsolutionone::Vector{Int}
  dsolution::Vector{Float64}
  zsolution::Vector{Float64}
  varzfixedsolutionzero::Vector{Int}
  restr::Vector{ConstraintRef}
 end 




#########################################################################
# Function optbestz used during DRL policy iteration cycle to find better 
#   first stage solution z

function optbestz(DESTC,NUMBERNODES)

#variables from Main
##first_layer_weights
##first_layer_biases
## second_layer_weights
##second_layer_biases

  V2= DESTC+ones(Int64,length(DESTC))
  V3 = union([1],V2)
  MHIGH = zeros(NUMBERNODES)
  MLOW = zeros(NUMBERNODES)
  LOW = 0
  UPPER = length(V2)+1
  L=zeros(NUMBERNODES,length(V2))
  U=zeros(NUMBERNODES,length(V2))

  for i in 1:NUMBERNODES
    for j in 1:length(V2)
      if first_layer_weights[j,i] < 0
        L[i,j]= UPPER
        U[i,j]= LOW
      else
        L[i,j]= LOW
        U[i,j]= UPPER
      end
    end
  end 
  
  for i  in 1:NUMBERNODES
    MHIGH[i]= sum(first_layer_weights[j,i]*U[i,j] for j in 1:length(V2))+ first_layer_biases[i]
    MLOW[i]= sum(first_layer_weights[j,i]*L[i,j] for j in 1:length(V2))+ first_layer_biases[i]
  end
 
  #println(MHIGH)
  #println(MLOW)
  #readline()

  model = Model(CPLEX.Optimizer)
  set_optimizer_attribute(model, "CPX_PARAM_MIPDISPLAY", 0)
  set_optimizer_attribute(model, "CPX_PARAM_SCRIND", 0)
  set_optimizer_attribute(model, "CPX_PARAM_TILIM", 10800)

  @variable(model, y[j in 1:NUMBERNODES]>=0)
  @variable(model, s[j in 1:NUMBERNODES],Bin)
  @variable(model, z[i in 1:length(V2)]>=1)
  @variable(model, x[i in V2,j in V2; i != j], Bin )
  @variable(model, obj>=0)

  @objective(model, Min, obj )

  @constraint(model,obj >= sum(second_layer_weights[i,1]*y[i] for i in 1:NUMBERNODES) + second_layer_biases[1] )

  @constraint(model, nn1[i in 1:NUMBERNODES], y[i] >= sum( first_layer_weights[j,i]*z[j] for j in 1:length(V2) )   + first_layer_biases[i] )

  @constraint(model, nn2[i in 1:NUMBERNODES], y[i] <= sum( first_layer_weights[j,i]*z[j] for j in 1:length(V2) )   + first_layer_biases[i]  - MLOW[i]*(1-s[i]) )
 
  @constraint(model, nn3[i in 1:NUMBERNODES], y[i] <= MHIGH[i]*s[i] )


  #No strengthen constraints

  @constraint(model, order1[i in V2,j in V2; i != j], x[i,j]+x[j,i]==1)
  @constraint(model, order2[i in V2,j in V2,k in V2 ; i != j && i != k && j != k], x[i,j]+x[j,k]+x[k,i]<=2)
  @constraint(model, order3[i in V2], z[findfirst(x -> x==i, V2)] == 1+sum(x[j,i] for j in V2 if i != j ))
    @constraint(model, order4[i in 1:length(V2)], z[i] <= length(V2) ) 
  #print(model)
  #readline()
  optimize!(model)
  if string(termination_status(model)) != "OPTIMAL"
     print(string(termination_status(model))," optbestz not optimal")
     readline()
  end
  return value.(z), objective_value(model), string(termination_status(model))

end #end optbestz function

#########################################################################
# Function rewardvariant2 used during DRL to calculate REWARD VARIANT 2 
# exact approach (optimal allocatio of available ODs) given a scenario and a first stage ordering

function rewardvariant2(V,A,REWV,PRICEOD,DESTC,scenario,firststage)

#variables from Main
## scenario
## firststage

  model = Model(CPLEX.Optimizer)
  V2= DESTC+ones(Int64,length(DESTC))
  firststage= firststage+ones(Int64,length(DESTC))
  V3 = union([1],V2)
  set_optimizer_attribute(model, "CPX_PARAM_MIPDISPLAY", 0)
  set_optimizer_attribute(model, "CPX_PARAM_SCRIND", 0)
  set_optimizer_attribute(model, "CPX_PARAM_TILIM", 10800)

  @variable(model, x[i in V3,j in V3; i != j], Bin)
  @variable(model, y[i in V3,j in V3; i != j] >= 0 )
  @variable(model, w[i in V2], Bin)
  @variable(model, v[i in V2], Bin)

  @objective(model, Min, sum(sum( REWV*A[i,j]*x[i,j]  for i  in V3 if i!=j) for j  in V3 ) 
+ sum( w[i]*PRICEOD[findfirst(x -> x==i, V2)] for i in V2))


  @constraint(model, flow1[i in V2, j in V3; i != j && scenario[findfirst(x -> x==i, V2)]==0], x[i,j]  == 0)

  @constraint(model, flow2[j in V2], sum(x[i,j]  for i in V3 if i != j) == v[j] )
  @constraint(model, flow3[i in V2], sum(x[i,j]  for j in V3 if i != j) == v[i] )

  @constraint(model, flow4, sum(x[i,1] for i in V2 ) - sum(x[1,i] for i in V2 ) == 0 )

  @constraint(model, rest1[i in V2], v[i]+w[i] <= 1)
  @constraint(model, rest2[i in V2], v[i]+w[i] >=  scenario[findfirst(x -> x==i, V2)])
  @constraint(model, rest3[i in V2], v[i] <=  scenario[findfirst(x -> x==i, V2)])
  @constraint(model, rest4[i in V2], w[i] <=  scenario[findfirst(x -> x==i, V2)])
  @constraint(model, rest5[i in V2], w[i] <=  scenario[length(V2)+findfirst(x -> x==i, V2)])

  @constraint(model, cap1[i in V2], sum(y[j,i] for j in V3  if i != j)- sum(y[i,j] for j in V3  if i != j) == v[i])

  @constraint(model, cap2, sum(y[1,j] for j in V2)== sum(v[j] for j in V2)  )
  @constraint(model, cap3[i in V2], y[i,1] ==0)
  
  @constraint(model, cap4[i in V3,j in V3; i != j], y[i,j] <= CAPV*x[i,j])

  #no time constraints now

  #respect order of firststage

  @constraint(model, order1[i in V2], 
sum(x[i,j] for j in V2 if (findfirst(x -> x==j, firststage) < findfirst(x -> x==i, firststage)    ) )== 0)
  @constraint(model, order2[i in V2], 
sum(x[j,i] for j in V2 if (findfirst(x -> x==j, firststage) < findfirst(x -> x==i, firststage)    ) ) <= findfirst(x -> x==i, firststage))
 
  optimize!(model)
  #println(sum(value.(w)),sum(scenario[length(V2)+1:end]))
  return sum(value.(w)),sum(scenario[length(V2)+1:end]),objective_value(model)
end  #end rewardvariant2 function


#########################################################################
# Function reoptimization used during simulation to find best route and OD allocation
# given a scenario


function reoptimization(V,A,REWV,PRICEOD,DESTC,scenario)

#variables from Main
## scenario
  
  V2= DESTC+ones(Int64,length(DESTC))
  V3 = union([1],V2)

  model = Model(CPLEX.Optimizer)
  set_optimizer_attribute(model, "CPX_PARAM_MIPDISPLAY", 0)
  set_optimizer_attribute(model, "CPX_PARAM_SCRIND", 0)
  set_optimizer_attribute(model, "CPX_PARAM_TILIM", 10800)

  @variable(model, x[i in V3,j in V3; i != j], Bin)
  @variable(model, y[i in V3,j in V3; i != j] >= 0 )

  @objective(model, Min, sum(sum( REWV*A[i,j]*x[i,j]  for i  in V3 if i!=j) for j  in V3 ) 
+ sum( (scenario[i]*scenario[i+length(V2)])*PRICEOD[i] for i in 1:length(V2)))

  @constraint(model, flow1[i in V2, j in V3; i != j && scenario[findfirst(x -> x==i, V2)]==0 ], x[i,j]  == 0)
  @constraint(model, flow11[i in V2, j in V3; i != j && scenario[findfirst(x -> x==i, V2)]==0 ], x[j,i]  == 0)
  @constraint(model, flow2[i in V2, j in V3; i != j && scenario[findfirst(x -> x==i, V2)]==1 && 
scenario[length(V2)+findfirst(x -> x==i, V2)]==1], x[i,j]  == 0) 
  @constraint(model, flow22[i in V2, j in V3; i != j && scenario[findfirst(x -> x==i, V2)]==1 && 
scenario[length(V2)+findfirst(x -> x==i, V2)]==1], x[j,i]  == 0) 

  @constraint(model, flow3[j in V2], sum(x[i,j]  for i in V3 if i != j) == 
scenario[findfirst(x -> x==j, V2)]*(1-scenario[findfirst(x -> x==j, V2)+length(V2)]) )
  @constraint(model, flow4[i in V2], sum(x[i,j]  for j in V3 if i != j) == 
scenario[findfirst(x -> x==i, V2)]*(1-scenario[findfirst(x -> x==i, V2)+length(V2)]) )

  @constraint(model, flow5, sum(x[i,1] for i in V2 ) - sum(x[1,i] for i in V2 ) == 0 )

  @constraint(model, cap1[i in V2], sum(y[j,i] for j in V3  if i != j)- sum(y[i,j] for j in V3  if i != j) == scenario[findfirst(x -> x==i, V2)]*(1-scenario[findfirst(x -> x==i, V2)+length(V2)]))

  @constraint(model, cap2, sum(y[1,j] for j in V2)== sum(scenario[1:length(V2)].*(ones(Int64,length(V2))-scenario[length(V2)+1:2*length(V2)]))  )
  @constraint(model, cap3[i in V2], y[i,1] ==0)
  
  @constraint(model, cap4[i in V3,j in V3; i != j], y[i,j] <= CAPV*x[i,j])
 

  #no time restrictions for now
 
  optimize!(model)
 
  return objective_value(model)
end  #end reoptimization function


########################################################################################
######## Set of functions to run DRO algorithm

function dro(V,A,DESTC,PROBC,PRICEOD,REWV,CAPV,PROBOD,REWOD)

function fillscenario(where, temp,size,position)
  if position == size+1
    push!(where,temp)
  else
    temp[position]=0
    fillscenario(where, temp,size,position+1)
    temp[position]=1
    fillscenario(where,temp,size,position+1)
  end
end


function troca(vetor, nodei, nodej)
  aux = vetor[nodei]
  vetor[nodei] = vetor[nodej]
  vetor[nodej] = aux;
end

function fillroute(vetor, inf, sup,route)
   
  if inf == sup
    data4 = copy(vetor)
    if data4[1] <= data4[end]
      push!(route,data4)
    end
  else
    for i = inf:sup
      troca(vetor, inf, i)
      fillroute(vetor, inf + 1, sup,route)
      troca(vetor, inf, i) #backtracking
    end
  end
end

function solveSTODDUEXACTN10(V,A,DESTC,PROBC,PRICEOD,REWV,CAPV,DURV,PROBOD,REWOD)
 
  function process(current)
    #Initialize cenarios and solutions to be added
    
    cenariosused = []
    routesused = []
    varxsolutionfixedzero = []
    varxsolutionfixedone = [] 
    vardsolutionfixedzero = []
    vardsolutionfixedone = []
    varzsolutionfixedzero = []
    restractive = []  
    node = current
    while true
      cenariosused = [cenariosused;tree[node].addedcenarios]
      routesused = [routesused;tree[node].addedsolutions]
      varxsolutionfixedzero = [varxsolutionfixedzero;tree[node].varxfixedsolutionzero]
      varxsolutionfixedone = [varxsolutionfixedone;tree[node].varxfixedsolutionone]
      vardsolutionfixedzero = [vardsolutionfixedzero;tree[node].vardfixedsolutionzero]
      vardsolutionfixedone = [vardsolutionfixedone;tree[node].vardfixedsolutionone]
      varzsolutionfixedzero = [varzsolutionfixedzero;tree[node].varzfixedsolutionzero]
      restractive = [restractive;tree[node].restr]
      node = tree[node].parent
      node == 0 && break
    end
 
    result = 0
    resultz= []
    Gomorycutsrounds = 0
    #Now formulate problem (relaxed problem now)
    #println("comecar node ",current," com routes ", length(routesused), " e cenarios", length(cenariosused))
    #read(STDIN,Char) 
    
    const4 = Vector{ConstraintRef}(undef,length(cenariosused))
    const11 = Vector{ConstraintRef}()
    const12 = Vector{ConstraintRef}(undef,length(scenarioall))
    
    z = Vector{VariableRef}(undef,length(routesused))
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPX_PARAM_SCRIND", 0)
    set_optimizer_attribute(model, "CPX_PARAM_TILIM", 10800)
    #Reserve first fixed indexes for continuous variables
    @variable(model,  s >= 0)
    @variable(model,  u[1:length(DESTC)]>=0)
    
   
    #println("varz fixed zero ",varzsolutionfixedzero)
    for r in 1:length(routesused)
      if in(routesused[r],varzsolutionfixedzero)
        z[r] = @variable(model, lower_bound = 0, upper_bound =0, base_name="z[$r]")
      else
        z[r] = @variable(model, lower_bound = 0, upper_bound =1, base_name="z[$r]")
      end
    end

    for i in 1:length(tree[current].zsolution)
       set_start_value(z[i],tree[current].zsolution[i])  #dual feasible?
    end
    println("p4")
    @variable(model,  x[i in 1:length(DESTC1),j in 1:length(DESTC1); i != j]>=0)

    for i in varxsolutionfixedzero
      set_lower_bound(x[i[1],i[2]],0)
      set_upper_bound(x[i[1],i[2]],0) 
    end
    for i in varxsolutionfixedone
      set_lower_bound(x[i[1],i[2]],1)
      set_upper_bound(x[i[1],i[2]],1)
    end
    
 
    @objective(model, Min, s - sum(PROBOD[i]*u[i] for i in 1:length(DESTC))  )

    for w in 1:length(cenariosused)
      const4[w] = @constraint(model,  s - sum(scenario[cenariosused[w]][i]*u[i] for i in 1:length(DESTC))  - sum(sum(sum(REWV*A[DESTC1[i],DESTC1[j]]*B7[routesused[r],findfirst(x -> x == scenario[cenariosused[w]], scenarioall),i,j]*z[r] for i in 1:length(DESTC1) if i != j) for j in 1:length(DESTC1)) for r in 1:length(routesused)) >= sum(PRICEOD[i]*scenario[cenariosused[w]][i] for i in 1:length(DESTC)))
    end

    @constraint(model, const5[i in 1:length(DESTC)], sum(B1[routesused[r],i]*z[r] for r in 1:length(routesused)) == 1)
      


    @constraint(model,  const10[i in 1:length(DESTC1),j in 1:length(DESTC1); i != j], x[i,j] - sum(B6[routesused[r],i,j]*z[r] for r in 1:length(routesused))==0)
 
    for sce in 1:length(scenarioall) #eliminate scenario all 0 and all 1
      if scenarioall[sce] != zeros(length(scenarioall[sce])) && scenarioall[sce] != ones(length(scenarioall[sce]))
        total = 0
        for k in 1:length(DESTC)
          if scenarioall[sce][k]== 1
            total += 1
          end
        end
        const12[sce]=@constraint(model, sum( sum(x[i,j] for i in 1:length(DESTC)  if i != j ) for j in 1:length(DESTC1))  >= ceil(total/CAPV) )
      else
        const12[sce]=@constraint(model, 0==0)
      end
    end


    const11 = [const11;restractive]
   
   
    while true
      optimize!(model)
      status = termination_status(model)
      result = objective_value(model)
      resultz = value.(z)
  
      stop = false
      mincost = false
      if status != :Optimal
        #prune   
        #pos = findfirst(x -> x == current, Queue)
        #deleteat!(Queue,pos)
        #delete!(Queue, current) 
        println("Infeasible...prune ", status)
        readline() 
        stop = true
        break 
      end

      global globalincumbentvalue
      global globalincumbentsolutiond
      global globalincumbentsolutionz
     
      
      xsol1 = getvalue(x)
      xsol = zeros(length(DESTC1),length(DESTC1))

      for i=1:length(DESTC1)
        for j=1:length(DESTC1)
          if i != j    
            if xsol1[i,j]<=0.00001 
              xsol[i,j] = 0
            elseif  xsol1[i,j] >=0.99999
              xsol[i,j]= 1
            else
              xsol[i,j]=xsol1[i,j]
            end
          end
        end
      end
      #for i=1:length(DESTC1)
      #  println(xsol[i,:])
      #end
      #read(STDIN,Char)

      bestr = 0
      E1dual = getdual(const5)
      G1dual = getdual(const10)
      D1dual = getdual(const4)
      maxcol = -Inf
      for r in 1:length(route)
        if !in(r,routesused)
          total= sum( B1[r,i]*E1dual[i] for i in 1:length(DESTC)) - sum(B6[r,i,j]*G1dual[i,j] for i in 1:length(DESTC1),j in 1:length(DESTC1) if  i != j) - sum(sum(sum(REWV*A[DESTC1[i],DESTC1[j]]*B7[r,findfirst(x -> x == scenario[cenariosused[w]], scenarioall),i,j]*D1dual[w] for i in 1:length(DESTC1) if i != j) for j in 1:length(DESTC1)) for w in 1:length(cenariosused) ) >= 0.00001  
          if total >= 0.00001
            if maxcol < total
              maxcol = total
              bestr = r
            end
          end
        end #r in route
      end
      
     
      if bestr != 0
        #println("time to insert route ",bestr," in node", current)
        r = bestr
        push!(routesused,r)
        push!(tree[current].addedsolutions,r)
        global ROUTCOUNT += 1
        mincost = true
        #read(STDIN,Char)
        tes = [B1[r,i] for i in 1:length(DESTC)]
        tes1 = [const5[i] for i in 1:length(DESTC) ]
        #for w in 1:length(cenariosused)   
        for i in 1:length(DESTC1),j in 1:length(DESTC1)
          if i !=j 
            push!(tes, -B6[r,i,j])
            push!(tes1, const10[i,j])
         end
        end
        for w in 1:length(cenariosused)
          push!(tes, -sum(sum(REWV*A[DESTC1[i],DESTC1[j]]*B7[r,findfirst(x -> x == scenario[cenariosused[w]], scenarioall),i,j] for i in 1:length(DESTC1) if i != j) for j in 1:length(DESTC1)) )
          push!(tes1, const4[w])
        end
        index = length(tree[current].addedsolutions)
        push!(z,@variable(model,  lower_bound = 0, upper_bound =1, base_name="z[$index]",objective = 0.0, inconstraints = tes1, coefficients = tes ))
        #setvalue(y[findfirst(x -> x == best_s, solutionsused)],1
      end
      

      #println("mincost =", mincost)
      if !mincost
        #println("after mincost not true no column inserted")
        if  result > globalincumbentvalue + 0.0001
          #prune  since a lower bound is already worse then incumbent 
          #pos = findfirst(x -> x == current, Queue)
          #deleteat!(Queue,pos)
          #delete!(Queue, current)
          #println("Prune by Bound =", result, ", ", globalincumbentvalue)
          #read(STDIN,Char) 
          stop = true
          break
        end

        stop2=false
        ssol = getvalue(s)
        usol= getvalue(u)
        if maximum(xsol-floor.(xsol)) <= 0.00001 
          #new integer solution
          #println("Integer Solution Found")
          ##############################################
          #Formulate Separation Problem and Solve 
          #Get First Stage results
          for i in 1:length(resultz) #check for error and display
            if resultz[i] > 0.0001 && resultz[i] < 0.99
              println("SOLUCAO NAO INTEIRA!! ",i, ",",resultz[i])
              read(STDIN,Char) 
            end
          end


          model2 = Model(CPLEX.Optimizer)
          set_optimizer_attribute(model, "CPX_PARAM_SCRIND", 0)
          set_optimizer_attribute(model, "CPX_PARAM_TILIM", 1000) #CplexSolver(CPX_PARAM_MIPDISPLAY = 0,CPX_PARAM_SCRIND=0,CPX_PARAM_TILIM=1000))
          @variable(model2, scen[1:length(DESTC)], Bin)
          @variable(model2, 0 <= y2[r in 1:length(routesused),i in 1:length(DESTC1), j in 1:length(DESTC1); i != j && resultz[r]> 0.001] <= 1) 
          @objective(model2, Min, ssol - sum(scen[i]*usol[i] for i in 1:length(DESTC))  - sum(sum(sum(REWV*A[DESTC1[i],DESTC1[j]]*y2[r,i,j] for i in 1:length(DESTC1) if i != j) for j in 1:length(DESTC1))   for r in 1:length(routesused) if resultz[r]> 0.001 ) - sum(PRICEOD[i]*scen[i] for i in 1:length(DESTC)))

      
          @constraint(model2, const21[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC);i != j && resultz[r] > 0.001], -(B2[routesused[r],i,j] + sum(scen[k]*B3[routesused[r],i,j,k]+B4[routesused[r],i,j,k]+B5[routesused[r],i,j,k] + 1-B1[routesused[r],k] for k in 1:length(DESTC) if k != j && k != i))*resultz[r] + y2[r,i,j] >= 1- scen[i] + 1 - scen[j] - length(DESTC))

          @constraint(model2, const22[r in 1:length(routesused),i in 1:length(DESTC); resultz[r] > 0.001], -(0 + sum(scen[k]*B2[routesused[r],k,i]+0+B2[routesused[r],i,k] + 1-B1[routesused[r],k] for k in 1:length(DESTC) if k != i))*resultz[r] + y2[r,length(DESTC1),i] >=  1 - scen[i] - length(DESTC) + 1)

          @constraint(model2, const23[r in 1:length(routesused),i in 1:length(DESTC); resultz[r] > 0.001], -(0 + sum(scen[k]*B2[routesused[r],i,k]+B2[routesused[r],k,i]+0 + 1-B1[routesused[r],k] for k in 1:length(DESTC) if k != i))*resultz[r] + y2[r,i,length(DESTC1)] >= 1- scen[i]  - length(DESTC) + 1) 


          @constraint(model2, const22b[r in 1:length(routesused),i in 1:length(DESTC); resultz[r] > 0.001], y2[r,length(DESTC1),i] <= (1-scen[i]) )
          @constraint(model2, const22c[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC); i!=j && resultz[r] > 0.001], y2[r,length(DESTC1),i] <= scen[j]*B2[routesused[r],j,i]+0+B2[routesused[r],i,j] + 1-B1[routesused[r],j] )

          @constraint(model2, const23b[r in 1:length(routesused),i in 1:length(DESTC); resultz[r] > 0.001], y2[r,i,length(DESTC1)] <= (1-scen[i]) )
          @constraint(model2, const23c[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC); i!=j && resultz[r] > 0.001], y2[r,i,length(DESTC1)] <= scen[j]*B2[routesused[r],i,j]+B2[routesused[r],j,i]+0 + 1-B1[routesused[r],j] )

          @constraint(model2, const21b[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC);i != j && resultz[r] > 0.001], y2[r,i,j] <= (1-scen[i]) )

          @constraint(model2, const21c[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC);i != j && resultz[r] > 0.001], y2[r,i,j] <= (1-scen[j])*B2[routesused[r],i,j] )

          @constraint(model2, const21d[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC),k in 1:length(DESTC);i != j && k!=i && k!=j && resultz[r] > 0.001], y2[r,i,j] <= scen[k]*B3[routesused[r],i,j,k]+B4[routesused[r],i,j,k]+B5[routesused[r],i,j,k] + 1-B1[routesused[r],k]  )

        
          #println("Will solve Separation Problem for integer solution")
           
          optimize!(model2)
 
          #println("solve model 2 ",getobjectivevalue(model2))
          #println(getvalue(scen))
          #read(STDIN,Char)
          if objective_value(model2) <= -0.0001 #-0.05
            #println("ADD NEW SCENARIO ",getvalue(scen), " get= ", getobjectivevalue(model2))
            #read(STDIN,Char)
            #Add new scenario and continue
            scenarionew = round.(value.(scen))
            pos = findfirst(x -> x == scenarionew, scenarioall)
            #println("pos= ", pos)
            #read(STDIN,Char)
            #println("Integer Solution not valid. New scenario: ", scenarionew)
            push!(scenario,scenarionew)
            #Update Current node information on scenarios used
            push!(tree[current].addedcenarios,size(scenario,1))
            push!(cenariosused,size(scenario,1))
            #Create new variables
  
            #Create new constraints
  
            push!(const4, @constraint(model,  s - sum(scenario[end][i]*u[i] for i in 1:length(DESTC))  - sum(sum(sum(REWV*A[DESTC1[i],DESTC1[j]]*B7[routesused[r],pos,i,j]*z[r] for i in 1:length(DESTC1) if i != j) for j in 1:length(DESTC1)) for r in 1:length(routesused)) >= sum(PRICEOD[i]*scenario[end][i] for i in 1:length(DESTC)) ))
            

            stop2=true
          else # end getobj < +.05
            #println("No new scenario on it solution")
            #end verification of scenario insertion for probable integer solution
            ##############################################
            #Verify if it can be new incubent
            if result < globalincumbentvalue
              #println("New Incumbent Found")
              globalincumbentvalue = result
              globalincumbentsolutiond = dsol 
              globalincumbentsolutionz = resultz
            end 
            #pos = findfirst(x -> x == current, Queue)
            #deleteat!(Queue,pos)
            #delete!(Queue, current)
            #println("Will break after integer solution")
            stop=true
            break
          end
        else  #maximum(xsol-floor.(xsol)) <= 0.00001 

          ########################START SEPARATION PROBLEM IF ROOT NODE
          if length(cenariosused) <= 30  && current == 1 
  
            model2 = Model(CPLEX.Optimizer)
            set_optimizer_attribute(model, "CPX_PARAM_SCRIND", 0)
            set_optimizer_attribute(model, "CPX_PARAM_TILIM", 1000) #CplexSolver(CPX_PARAM_MIPDISPLAY = 0,CPX_PARAM_SCRIND=0,CPX_PARAM_TILIM=1000))

            @variable(model2, scen[1:length(DESTC)], Bin)
            @variable(model2, 0 <= y2[r in 1:length(routesused),i in 1:length(DESTC1), j in 1:length(DESTC1); i != j && resultz[r]> 0] <= 1) 
 
            @objective(model2, Min, ssol - sum(scen[i]*usol[i] for i in 1:length(DESTC)) - sum(sum(sum(REWV*A[DESTC1[i],DESTC1[j]]*y2[r,i,j] for i in 1:length(DESTC1) if i != j) for j in 1:length(DESTC1)) for r in 1:length(routesused) if resultz[r]> 0 ) - sum(PRICEOD[i]*scen[i] for i in 1:length(DESTC)))

      
            @constraint(model2, const21[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC);i != j && resultz[r] > 0], -(B2[routesused[r],i,j] + sum(scen[k]*B3[routesused[r],i,j,k]+B4[routesused[r],i,j,k]+B5[routesused[r],i,j,k] + 1-B1[routesused[r],k] for k in 1:length(DESTC) if k != j && k != i))*resultz[r] + y2[r,i,j] >= 1- scen[i] + 1 - scen[j] - length(DESTC))

            @constraint(model2, const22[r in 1:length(routesused),i in 1:length(DESTC); resultz[r] > 0], -(0 + sum(scen[k]*B2[routesused[r],k,i]+0+B2[routesused[r],i,k] + 1-B1[routesused[r],k] for k in 1:length(DESTC) if k != i))*resultz[r] + y2[r,length(DESTC1),i] >=  1 - scen[i] - length(DESTC) + 1)

            @constraint(model2, const23[r in 1:length(routesused),i in 1:length(DESTC); resultz[r] > 0], -(0 + sum(scen[k]*B2[routesused[r],i,k]+B2[routesused[r],k,i]+0 + 1-B1[routesused[r],k] for k in 1:length(DESTC) if k != i))*resultz[r] + y2[r,i,length(DESTC1)] >= 1- scen[i]  - length(DESTC) + 1) 


            @constraint(model2, const22b[r in 1:length(routesused),i in 1:length(DESTC); resultz[r] > 0], y2[r,length(DESTC1),i] <= (1-scen[i]) )
            @constraint(model2, const22bb[r in 1:length(routesused),i in 1:length(DESTC); resultz[r] > 0], y2[r,length(DESTC1),i] <= resultz[r] )
            @constraint(model2, const22c[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC); i!=j && resultz[r] >0], y2[r,length(DESTC1),i] <= scen[j]*B2[routesused[r],j,i]+0+B2[routesused[r],i,j] + 1-B1[routesused[r],j] )

            @constraint(model2, const23b[r in 1:length(routesused),i in 1:length(DESTC); resultz[r] > 0], y2[r,i,length(DESTC1)] <= (1-scen[i]) )
            @constraint(model2, const23bb[r in 1:length(routesused),i in 1:length(DESTC); resultz[r] > 0], y2[r,i,length(DESTC1)] <= resultz[r] )
            @constraint(model2, const23c[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC); i!=j && resultz[r] > 0], y2[r,i,length(DESTC1)] <= scen[j]*B2[routesused[r],i,j]+B2[routesused[r],j,i]+0 + 1-B1[routesused[r],j] )

            @constraint(model2, const21b[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC);i != j && resultz[r] > 0], y2[r,i,j] <= (1-scen[i]) )
            @constraint(model2, const21bb[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC);i != j && resultz[r] > 0], y2[r,i,j] <= resultz[r] )

            @constraint(model2, const21c[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC);i != j && resultz[r] > 0], y2[r,i,j] <= (1-scen[j])*B2[routesused[r],i,j] )

            @constraint(model2, const21d[r in 1:length(routesused),i in 1:length(DESTC),j in 1:length(DESTC),k in 1:length(DESTC);i != j && k!=i && k!=j && resultz[r] > 0], y2[r,i,j] <= scen[k]*B3[routesused[r],i,j,k]+B4[routesused[r],i,j,k]+B5[routesused[r],i,j,k] + 1-B1[routesused[r],k]  )

        

           
            optimize!(model2)

 
            #println("solve model 2 ",getobjectivevalue(model2))
            #println(getvalue(scen))
            #read(STDIN,Char)
            if  objective_value(model2) <= -0.0001 #-0.05
              #println(getvalue(scen), " get= ", getobjectivevalue(model2))
              #read(STDIN,Char)
              #Add new scenario and continue
              scenarionew = round.(value.(scen))
              pos = findfirst(x -> x == scenarionew, scenarioall)
              #println("pos= ", pos)
              #read(STDIN,Char)
              #println("Integer Solution not valid. New scenario: ", scenarionew)
              push!(scenario,scenarionew)
              #Update Current node information on scenarios used
              push!(tree[current].addedcenarios,size(scenario,1))
              push!(cenariosused,size(scenario,1))
              #Create new variables
  
              #Create new constraints

              push!(const4, @constraint(model,  s - sum(scenario[end][i]*u[i] for i in 1:length(DESTC))  - sum(sum(sum(REWV*A[DESTC1[i],DESTC1[j]]*B7[routesused[r],pos,i,j]*z[r] for i in 1:length(DESTC1) if i != j) for j in 1:length(DESTC1)) for r in 1:length(routesused)) >= sum(PRICEOD[i]*scenario[end][i] for i in 1:length(DESTC)) ))
              stop2=true
            end
          end #end current == 1

          zdual = getdual(z)
          for i in 1:length(zdual)
            if result + zdual[i] > globalincumbentvalue && !in(i,varzsolutionfixedzero)
              #println("node ",current," z solution for route ",routesused[i], " will not improve")
              #read(STDIN,Char)
              #if resultz[i] >= 0.1 && resultz[i] <= 0.9
              #  println("erro z[", i,"]")
              #  read(STDIN,Char)
              #end
              set_lower_bound(z[i],0)
              set_upper_bound(z[i],0) 
              push!(varzsolutionfixedzero,routesused[i])
              push!(tree[current].varzfixedsolutionzero,routesused[i])
              #Should still eliminate constraints not needed here  
            end
          end
          #End Section for reduced Cost variable fixing

          if !stop2  #time for branching
            #println("time for branching")
            if maximum(xsol-floor.(xsol)) > 0.00001
              if length(vardsolutionfixedzero) != 0 || length(vardsolutionfixedone) != 0
                #println("d ja estava fixed e veio xij frac")
                #println(vardsolutionfixedzero)
                #println(vardsolutionfixedone)
                #read(STDIN,Char)
              end 
              a = maximum(xsol-floor.(xsol))
              f=0
              g=0
              for i in 1:length(DESTC1),j in 1:length(DESTC1)
                if i != j
                  if xsol[i,j]-floor.(xsol[i,j])== a
                    f = i
                    g = j
                    break
                  end
                end
              end
              #println("Sol x still frac, so split into two nodes using max frac", f," ,",g) 
              b =[f;g] 
              push!(tree,TreeNode(current,[],[],[],[],[b],[],[],dsol,resultz,[],const11))
              push!(tree[current].children,length(tree))
              #push!(Queue,length(tree))
              Queue[length(tree)]=result


              push!(tree,TreeNode(current,[],[],[],[b],[],[],[],dsol,resultz,[],const11))
              #current = length(tree)
              push!(tree[current].children,length(tree))
              ##push!(Queue,length(tree))
              Queue[length(tree)]=result

              ##pos = findfirst(x -> x == current, Queue)
              ##deleteat!(Queue,pos)
              ##delete!(Queue,current)

              ##current=Queue[end]
              stop = true
              ##Gomorycutsrounds = 0
              ##current=length(tree)

              #global NODECOUNT += 1
              #set_lower_bound(x[f,g],0)
              #set_upper_bound(x[f,g],0)
              #push!(varxsolutionfixedzero,b) 
  
            else
              println("situation not covered")
              read(STDIN,Char)
            end
          end #stop2
        end
      end # !mincost
      stop && break
    end #end while true 'solve)
    return model, result 
  end #end Process function

  
  #########################################################################
  # MAIN PROGRAM of function DRO 
  #Generate Prices, Routes and Scenarios to be used
  ###############################################
  #Create vectors for all possible scenarios
  global scenarioall = Vector{Vector{Int}}() 
  fillscenario(scenarioall,zeros(length(DESTC)),length(DESTC),1)
  println("All possible Scenario vectors created of size ", length(scenarioall) )
  #Create vectors for all possible routings
  global route = Vector{Vector{Int}}() 
  for s in 1:length(scenarioall)
    if sum(scenarioall[s]) <= CAPV
      groupe = []
      for c in 1:length(scenarioall[s])
        if scenarioall[s][c] !=0
          push!(groupe,c)
        end
      end
      if groupe != []
        fillroute(groupe, 1, length(groupe),route)  #symetry breaking included
        #fillroute(collect(1:length(DESTC)), 1, length(DESTC),route)
      end
    end
  end
  println("Route vectors created of size ", length(route))
  #println(route)
  #readline()
  #################################################

  ###############################################
  #Re-Create vectors for nly initial scenarios - use col  cut generation afterwards
  global scenario = Vector{Vector{Int}}() 
  #Create  initial scenarios
  push!(scenario, zeros(length(DESTC)))   #Scenario 1 has to be all zeros. All customers present. Scenario 1 is used for calculations later. 
  for i = 1:length(DESTC)
    SAMPLETEST = zeros(length(DESTC))
    SAMPLETEST[i]=1
    push!(scenario, SAMPLETEST)
  end
  push!(scenario, ones(length(DESTC)))
  ################################################

  #scenario = scenarioall
  scenariosused = collect(1:length(scenario))

  #Bypass REWOD and define new prices to pay ODs
  DESTC1=union(DESTC,[1])
  ################################################
  #Calculate needed parameters for routes and store:
  
  global B1 = zeros(length(route),length(DESTC))
  global B2 = zeros(length(route),length(DESTC1),length(DESTC1))
  global B3 = zeros(length(route),length(DESTC),length(DESTC),length(DESTC))
  global B4 = zeros(length(route),length(DESTC),length(DESTC),length(DESTC))
  global B5 = zeros(length(route),length(DESTC),length(DESTC),length(DESTC))
  global B6 = zeros(length(route),length(DESTC1),length(DESTC1)) 
  global B7 = zeros(length(route),length(scenarioall),length(DESTC1),length(DESTC1))  
  for r in 1:length(route)
    
    B6[r,route[r][end],length(DESTC1)] = 1
    B6[r,length(DESTC1),route[r][1]] = 1
    #println("route= ",route[r])
    #read(STDIN,Char)
    
    for i in 1:length(DESTC)
      
      B2[r,length(DESTC1),i] = 1
      B2[r,i,length(DESTC1)] = 1
      if findfirst(x -> x==i, route[r]) != nothing
        if i != route[r][end]
          B6[r,i,route[r][findfirst(x -> x==i, route[r])+1]] = 1
          #print("B6[ ",r," ,",i," ,",route[r][findfirst(x -> x==i, route[r])+1]," ]= ",B6[r,i,route[r][findfirst(x -> x==i, route[r])+1]], "; ",route[r])
        end
        B1[r,i] = 1
        #println("B1[ ",r,",",i," ]= ", B1[r,i])
        #read(STDIN,Char)
      end
       
      for j in 1:length(DESTC)
        if  findfirst(x -> x==i, route[r]) != nothing && findfirst(x -> x==j, route[r]) != nothing && (findfirst(x -> x==j, route[r]) > findfirst(x -> x==i, route[r]))
           B2[r,i,j] = 1
           #print("B2[ ",r," ,",i," ,",j," ]= ",B2[r,i,j], "; ")
           #read(STDIN,Char)
        end
        
        #println()
        for k in 1:length(DESTC)
          if i!=j && j!=k && k!=i &&findfirst(x -> x==i, route[r]) != nothing && findfirst(x -> x==j, route[r]) != nothing && findfirst(x -> x==k, route[r]) != nothing && findfirst(x -> x==k, route[r]) > findfirst(x -> x==i, route[r]) && findfirst(x -> x==k, route[r]) < findfirst(x -> x==j, route[r])
            B3[r,i,j,k] = 1
            #print("B3[ ",r," ,",i,",",j," ,",k," ]= ",B3[r,i,j,k]," ;")
            #read(STDIN,Char)
          end
          
          #println()
          if i!=j && j!=k && k!=i &&findfirst(x -> x==i, route[r]) != nothing && findfirst(x -> x==j, route[r]) != nothing && findfirst(x -> x==k, route[r]) != nothing && findfirst(x -> x==k, route[r]) < findfirst(x -> x==i, route[r]) && findfirst(x -> x==k, route[r]) < findfirst(x -> x==j, route[r])
            B4[r,i,j,k] = 1
            #print("B4[ ",r," ,",i,",",j," ,",k," ]= ",B4[r,i,j,k]," ;")
            #read(STDIN,Char)
            
          end
          #println()
          if i!=j && j!=k && k!=i &&findfirst(x -> x==i, route[r]) != nothing && findfirst(x -> x==j, route[r]) != nothing && findfirst(x -> x==k, route[r]) != nothing && findfirst(x -> x==k, route[r]) > findfirst(x -> x==i, route[r]) && findfirst(x -> x==k, route[r]) > findfirst(x -> x==j, route[r]) 
            B5[r,i,j,k] = 1
            #print("B5[ ",r," ,",i,",",j," ,",k," ]= ",B5[r,i,j,k]," ;")
            #read(STDIN,Char)
          end
          
          #println()
        end
      end
    end
  end
  
  for r in 1:length(route), w in 1:length(scenarioall),i in 1:length(DESTC),j in 1:length(DESTC)
    if i != j
        B7[r,w,i,j]  = (1- scenarioall[w][i])*(1 - scenarioall[w][j])*B2[r,i,j]*prod(scenarioall[w][k]*B3[r,i,j,k]+B4[r,i,j,k]+B5[r,i,j,k] + 1-B1[r,k] for k in 1:length(DESTC) if k != j && k != i)
    end
  end
  
  for r in 1:length(route), w in 1:length(scenarioall),i in 1:length(DESTC)
    B7[r,w,length(DESTC1),i ]  = (1 - scenarioall[w][i])*prod(scenarioall[w][k]*B2[r,k,i]+B2[r,i,k] + 1-B1[r,k] for k in 1:length(DESTC) if k != i)  
    B7[r,w,i,length(DESTC1)] = ( 1- scenarioall[w][i] )*prod(scenarioall[w][k]*B2[r,i,k]+B2[r,k,i] + 1-B1[r,k] for k in 1:length(DESTC) if k != i) 
  end 
  #####################################################################
  #read(STDIN,Char)
 
  #Initialize routes and compensations to be added. With the lack of better heuristic for incubent solutions just defined ad-hoc routes and compensations now 
  routesused = []
  for r in 1:length(route)
    if length(route[r]) <= 1
       push!(routesused,r)
    end
  end
  
  #routesused = collect(1:length(route))
  #Initialize branch-and-bound tree and queue
  #Queue = Vector{Int}()
  Queue = PriorityQueue()
  tree = Vector{TreeNode}()
  push!(tree,TreeNode(0,[],routesused,scenariosused,[],[],[],[],[],[],[],[])) 
  #push!(Queue,1)
  Queue[1]= 0
  #Start processing queue in a Deep First mode
  global heursol
  global globalincumbentvalue = +Inf
  global globalincumbentsolutiond = []
  global globalincumbentsolutionz = []
    
  masterx = 0
  master = []
  #always process last element of queue
  while length(Queue)>0
    #current=Queue[end]
    
    current= dequeue!(Queue)
    
    (master, masterx) =process(current)
    #if current == 1
    #  global ROOTSOL=masterx
    #  #global ROOTTIME= TimeMain +TimePric+TimeSce
    #end
    
    #global NODECOUNT += 1
    if length(Queue) > 0 && globalincumbentvalue != +Inf
      a,b=peek(Queue)
      if (globalincumbentvalue-b)/globalincumbentvalue < 0.02
        break 
      end
    end
    #if TimeMain +TimePric+TimeSce >= 10800+3600
    #  break
    #end
    
  end
  
  for i in 1:length(globalincumbentsolutionz)
    if globalincumbentsolutionz[i] > 0.0001 && globalincumbentsolutionz[i] < 0.99
      println("SOLUCAO NAO INTEIRA!! ",i)
      readline()
    end
  end
  #println("result= ",globalincumbentvalue,"dsol= ", globalincumbentsolutiond, "time = ", TimeMain +TimePric+TimeSce)
  #global NROUTES = sum(globalincumbentsolutionz)
  #global PERHIGH = sum(globalincumbentsolutiond)/length(globalincumbentsolutiond)
  return globalincumbentvalue,globalincumbentsolutiond
end #end N10 function


#No time constrainst
result,initialsol= solveSTODDUEXACTN10(V,A,DESTC,PROBC,PRICEOD,REWV,CAPV,DURV,PROBOD,REWOD)

return initialsol

end #end function DRO



