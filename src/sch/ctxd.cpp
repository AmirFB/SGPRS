# include <ctxd.hpp>

# include <ctx.hpp>

using namespace FGPRS;

ContextData::ContextData() {}

ContextData::ContextData(MyContext* context)
{
	this->context = context;
	smCount = context->smCount;
	isolatedExecutionTime = 0;
	occupiedExecutionTime = 0;
}

ContextData::ContextData(MyContext* context, double isolatedExecutionTime, double occupiedExecutionTime)
{
	this->context = context;
	smCount = context->smCount;
	this->isolatedExecutionTime = isolatedExecutionTime;
	this->occupiedExecutionTime = occupiedExecutionTime;
	this->isolatedExecutionTimeExp = exp(isolatedExecutionTime);
	this->occupiedExecutionTimeExp = exp(occupiedExecutionTime);
}

void ContextData::stackExecutionTime(ContextData ctxData)
{
	this->isolatedExecutionTime += ctxData.isolatedExecutionTime;
	this->occupiedExecutionTime += ctxData.occupiedExecutionTime;
	this->isolatedExecutionTimeExp += exp(ctxData.isolatedExecutionTime);
	this->occupiedExecutionTimeExp += exp(ctxData.occupiedExecutionTime);
}