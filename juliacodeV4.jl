#########################################################################
# Julia Code to be used integrated with Python for 
# Deep Reinforcement Learning Algorithm for Last mile delivery with crowd shipping
# 

#using JuMP, CPLEX, Combinatorics, Dates, Statistics, MathProgBase, DataStructures 
using JuMP, CPLEX, MathProgBase, Random

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
# Function rewardvariant2 used during DRL to calculate REWARD VARIANT 2 
# exact approach (optimal allocation of available ODs) given a scenario and a first stage ordering

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





