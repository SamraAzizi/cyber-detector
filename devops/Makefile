.PHONY: build up down logs clean

build:
	docker-compose -f devops/docker-compose.yml build

up:
	docker-compose -f devops/docker-compose.yml up -d

down:
	docker-compose -f devops/docker-compose.yml down

logs:
	docker-compose -f devops/docker-compose.yml logs -f

clean:
	docker-compose -f devops/docker-compose.yml down -v
	docker system prune -f

migrate:
	docker-compose -f devops/docker-compose.yml exec backend python manage.py migrate

train-model:
	docker-compose -f devops/docker-compose.yml run --rm ml python ml/training/train_model.py